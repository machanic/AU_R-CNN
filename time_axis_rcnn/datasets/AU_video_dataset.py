import cv2
import random

import chainer
import numpy as np
import os
import config
from collections_toolkit.ordered_default_dict import DefaultOrderedDict
from img_toolkit.face_mask_cropper import FaceMaskCropper
from time_axis_rcnn.datasets.AU_dataset import AUDataset
import random

class AU_video_dataset(chainer.dataset.DatasetMixin):

    def __init__(self, au_image_dataset:AUDataset,
                  sample_frame=10,  paper_report_label_idx=None
                ):
        self.AU_image_dataset = au_image_dataset
        self.sample_frame = sample_frame
        self.paper_report_label_idx = paper_report_label_idx


        self.seq_dict = DefaultOrderedDict(list)
        for idx, (img_path, *_) in enumerate(self.AU_image_dataset.result_data):
            video_seq_id = self.extract_sequence_key(img_path)
            self.seq_dict[video_seq_id].append(idx)

        self._order = None
        self.offsets = []
        self.result_data = []
        self.reset()

    def __len__(self):
        return len(self.result_data)

    def reset(self):
        self.offsets.clear()
        T = self.sample_frame
        jump_frame = 1  # just no jump
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
        else:
            for sequence_id, fetch_idx_lst in self.seq_dict.items():
                self.offsets.append(fetch_idx_lst)
        previous_data_length = len(self.result_data)
        self.result_data.clear()
        for fetch_idx_list in self.offsets:
            if len(fetch_idx_list) < T:  # RGB batch cannot pad!
                rest_pad_len = T - len(fetch_idx_list)
                fetch_idx_list = np.pad(fetch_idx_list, (0, rest_pad_len), 'edge')
            assert len(fetch_idx_list) == T
            for fetch_id in fetch_idx_list:
                self.result_data.append(self.AU_image_dataset.result_data[fetch_id])

        if previous_data_length != 0:
            if previous_data_length < len(self.result_data):
                assert (len(self.result_data) - previous_data_length) % T == 0
                del self.result_data[previous_data_length - len(self.result_data): ]
            elif previous_data_length > len(self.result_data):
                assert len(self.result_data) % T == 0
                assert (previous_data_length - len(self.result_data)) % T == 0
                chunks = [self.result_data[i:i + T] for i in range(0, len(self.result_data), T)]
                while previous_data_length > len(self.result_data):
                    self.result_data.extend(random.choice(chunks))
            assert len(self.result_data) == previous_data_length


    def get_example(self, i):
        rgb_path, flow_path, AU_set, database_name = self.result_data[i]
        video_seq_id = self.extract_sequence_key(rgb_path)
        # note that batch now is mix of T and batch_size, we must be reshape later
        try:
            rgb_face, flow_face, bbox, label = self.AU_image_dataset.get_from_entry(flow_path, rgb_path, AU_set, database_name)
            assert bbox.shape[0] == label.shape[0]
            if bbox.shape[0] != config.BOX_NUM[database_name]:
                print("found one error image: {0} box_number:{1}".format(rgb_path, bbox.shape[0]))
                bbox = bbox.tolist()
                label = label.tolist()

                if len(bbox) > config.BOX_NUM[database_name]:
                    all_del_idx = []
                    for idx, box in enumerate(bbox):
                        if FaceMaskCropper.calculate_area(*box) / float(config.IMG_SIZE[0] * config.IMG_SIZE[1]) < 0.01:
                            all_del_idx.append(idx)
                    for del_idx in all_del_idx:
                        del bbox[del_idx]
                        del label[del_idx]

                while len(bbox) < config.BOX_NUM[database_name]:
                    index = 0
                    bbox.insert(0, bbox[index])
                    label.insert(0, label[index])
                while len(bbox) > config.BOX_NUM[database_name]:
                    del bbox[-1]
                    del label[-1]

                bbox = np.stack(bbox)
                label = np.stack(label)
        except IndexError:
            print("image path : {} not get box".format(rgb_path))
            label = np.zeros(len(config.AU_SQUEEZE), dtype=np.int32)
            for AU in AU_set:
                np.put(label, config.AU_SQUEEZE.inv[AU], 1)
            if self.paper_report_label_idx:
                label = label[self.paper_report_label_idx]

            rgb_whole_image = np.transpose(cv2.resize(cv2.imread(rgb_path), config.IMG_SIZE), (2, 0, 1))
            flow_whole_image = np.transpose(cv2.resize(cv2.imread(flow_path), config.IMG_SIZE), (2, 0, 1))
            whole_bbox = np.tile(np.array([1, 1, config.IMG_SIZE[1] - 2, config.IMG_SIZE[0] - 2], dtype=np.float32),
                                 (config.BOX_NUM[database_name], 1))
            whole_label = np.tile(label, (config.BOX_NUM[database_name], 1))
            return rgb_whole_image, flow_whole_image, whole_bbox, whole_label, rgb_path

        assert bbox.shape[0] == config.BOX_NUM[database_name], bbox.shape[0]
        if self.paper_report_label_idx:
            label = label[:, self.paper_report_label_idx]
        return rgb_face, flow_face, bbox, label, rgb_path


    def extract_sequence_key(self, img_path):
        return "/".join((img_path.split("/")[-3], img_path.split("/")[-2]))

