import cv2
import random

import chainer
import numpy as np

import config
from collections_toolkit.ordered_default_dict import DefaultOrderedDict
from space_time_AU_rcnn.datasets.AU_dataset import AUDataset


class AU_video_dataset(chainer.dataset.DatasetMixin):

    def __init__(self, au_image_dataset:AUDataset,
                  sample_frame=10, train_mode=True, paper_report_label_idx=None, fetch_use_parrallel_iterator=True):
        self.AU_image_dataset = au_image_dataset
        self.sample_frame = sample_frame
        self.train_mode = train_mode
        self.paper_report_label_idx = paper_report_label_idx
        self.class_num = len(config.AU_SQUEEZE)
        if self.paper_report_label_idx:
            self.class_num = len(self.paper_report_label_idx)

        self.seq_dict = DefaultOrderedDict(list)
        for idx, (img_path, *_) in enumerate(self.AU_image_dataset.result_data):
            video_seq_id = self.extract_sequence_key(img_path)
            self.seq_dict[video_seq_id].append(idx)

        self.fetch_use_parrallel_iterator = fetch_use_parrallel_iterator
        if fetch_use_parrallel_iterator:
            self._order = None
            self.offsets = []
            self.result_data = []
            self.reset()


    def __len__(self):
        if self.fetch_use_parrallel_iterator:
            return len(self.result_data)
        return len(self.offsets)

    def reset(self):
        self.offsets.clear()
        T = self.sample_frame
        jump_frame = random.randint(4, 7)
        if T > 0:
            for sequence_id, fetch_idx_lst in self.seq_dict.items():
                for start_offset in range(jump_frame):
                    sub_idx_list = fetch_idx_lst[start_offset::jump_frame]
                    for i in range(0, len(sub_idx_list), T):
                        extended_list = sub_idx_list[i: i + T]
                        if len(extended_list) < T:
                            last_idx = extended_list[-1]
                            rest_list = list(filter(lambda e: e > last_idx, fetch_idx_lst))
                            if len(rest_list) > T - len(extended_list):
                                extended_list.extend(sorted(random.sample(rest_list, T - len(extended_list))))
                            else:
                                extended_list.extend(sorted(rest_list))
                        self.offsets.append(extended_list)

                # self.offsets.extend(
                #     [fetch_idx_lst[i:i + T] for i in range(0, len(fetch_idx_lst), T)])
        else:
            for sequence_id, fetch_idx_lst in self.seq_dict.items():
                self.offsets.append(fetch_idx_lst)

        self._order = np.random.permutation(len(self.offsets))
        self.result_data.clear()
        for order in self._order:
            fetch_idx_list = self.offsets[order]

            if len(fetch_idx_list) < T:
                rest_pad_len = T - len(fetch_idx_list)
                fetch_idx_list = np.pad(fetch_idx_list, (0, rest_pad_len), 'edge')
            assert len(fetch_idx_list) == T
            for fetch_id in fetch_idx_list:
                self.result_data.append(self.AU_image_dataset.result_data[fetch_id])

    def get_example(self, i):
        if self.fetch_use_parrallel_iterator:
            return self.parallel_get_example(i)
        return self.serial_get_example(i)

    def extract_sequence_key(self, img_path):
        return "/".join((img_path.split("/")[-3], img_path.split("/")[-2]))

    # we must set shuffle = False in this situation
    def parallel_get_example(self, i):
        img_path, AU_set, database_name = self.result_data[i]
        # note that batch now is mix of T and batch_size, we must be reshape later
        try:
            cropped_face, bbox, label = self.AU_image_dataset.get_from_entry(img_path, AU_set, database_name)
            assert bbox.shape[0] == label.shape[0]
            if bbox.shape[0] != config.BOX_NUM[database_name]:
                print("found one error image: {0} box_number:{1}".format(img_path, bbox.shape[0]))
                bbox = bbox.tolist()
                label = label.tolist()
                while len(bbox) < config.BOX_NUM[database_name]:
                    index = random.randint(0, len(bbox) - 1)
                    bbox.append(bbox[index])
                    label.append(label[index])
                while len(bbox) > config.BOX_NUM[database_name]:
                    bbox.pop(0)
                    label.pop(0)
                bbox = np.stack(bbox)
                label = np.stack(label)
        except IndexError:
            print("image path : {} not get box".format(img_path))
            label = np.zeros(len(config.AU_SQUEEZE), dtype=np.int32)
            for AU in AU_set:
                np.put(label, config.AU_SQUEEZE.inv[AU], 1)
            if self.paper_report_label_idx:
                label = label[self.paper_report_label_idx]
            return np.transpose(cv2.resize(cv2.imread(img_path), config.IMG_SIZE), (2,0,1)), \
                   np.tile(np.array([0, 0, config.IMG_SIZE[1]-1, config.IMG_SIZE[0]-1], dtype=np.float32),
                           (config.BOX_NUM[database_name], 1)), \
                   np.tile(label, (config.BOX_NUM[database_name], 1))

        assert bbox.shape[0] == config.BOX_NUM[database_name], bbox.shape[0]
        if self.paper_report_label_idx:
            label = label[:, self.paper_report_label_idx]
        return cropped_face, bbox, label


    def serial_get_example(self, i):
        if i > len(self):
            raise IndexError("Index too large , i = {}".format(i))
        fetch_img_idx_list = self.offsets[self._order[i]]
        sequence_images = []
        sequence_boxes = []
        sequence_labels = []
        if self.train_mode and 0 < self.sample_frame < len(fetch_img_idx_list):
            choice_frame = np.random.choice(np.arange(len(fetch_img_idx_list)), size=self.sample_frame, replace=False)
            choice_frame = np.sort(choice_frame)
            fetch_img_idx_list = [fetch_img_idx_list[frame] for frame in choice_frame]

        for img_idx in fetch_img_idx_list:
            try:
                cropped_face, bbox, label = self.AU_image_dataset[img_idx]
            except IndexError:
                print("error image_path: {} not fetch!".format(self.AU_image_dataset.result_data[img_idx][0]))
                continue
            assert cropped_face is not None
            assert len(bbox) == len(label)
            if bbox.shape[0] != config.BOX_NUM[self.AU_image_dataset.database]:
                print("error! image: {0} box number is {1} != {2}".format(self.AU_image_dataset.result_data[img_idx][0],
                                                                          bbox.shape[0],
                                                                          config.BOX_NUM[self.AU_image_dataset.database]))
                continue
            sequence_images.append(cropped_face)
            sequence_boxes.append(bbox)
            sequence_labels.append(label)
        if sequence_images:
            sequence_images = np.stack(sequence_images)  # T, C, H, W
            sequence_boxes = np.stack(sequence_boxes)  # T, R, 4
            sequence_labels = np.stack(sequence_labels)  # T, R, 22/12
        else:
            sequence_images = np.expand_dims(cropped_face, 0)
            sequence_boxes = np.expand_dims(bbox, 0)
            sequence_labels = np.stack(label, 0)

        if self.paper_report_label_idx:
            sequence_labels = sequence_labels[:, :, self.paper_report_label_idx]

        return sequence_images, sequence_boxes, sequence_labels
