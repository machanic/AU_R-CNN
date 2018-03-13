import cv2

from multiprocessing.pool import Pool
from multiprocessing import Process

from img_toolkit.face_mask_cropper import FaceMaskCropper
from space_time_AU_rcnn.datasets.AU_dataset import AUDataset
import numpy as np
from space_time_AU_rcnn.datasets.parallel_tools import parallel_landmark_and_conn_component, pack_function_for_map

import config
from collections import defaultdict
from collections_toolkit.ordered_default_dict import DefaultOrderedDict
import random


class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(Pool):
    Process = NoDaemonProcess


class AUTimeSequenceDataset(AUDataset):


    def __init__(self, database, fold, split_name, split_index, mc_manager, train_all_data,
                  previous_frame=50, sample_frame=25, train_mode=True, paper_report_label_idx=None, fetch_mode=1,
                 shuffle_T=False):
        super(AUTimeSequenceDataset, self).__init__(database, fold, split_name, split_index, mc_manager, train_all_data)
        self.previous_frame = previous_frame
        self.sample_frame = sample_frame
        self.train_mode = train_mode
        self.paper_report_label_idx = paper_report_label_idx
        self.shuffle_T = shuffle_T
        if fetch_mode > 1:
            self.fetch_func = self.get_parallel_example
        else:
            self.fetch_func = self.get_nonparallel_example

    def extract_sequence_key(self, img_path):
        return "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))

    def get_nonparallel_example(self, fetch_list):
        for i in fetch_list:
            if i > len(self.result_data):
                raise IndexError("Index too large")
            yield super(AUTimeSequenceDataset, self).get_example(i)

    def get_parallel_example(self, fetch_list):
        parallel_data = []
        img_path_label_dict = dict()
        for i in fetch_list:
            if i > len(self.result_data):
                raise IndexError("Index too large")
            img_path, AU_set, database_name = self.result_data[i]
            img_path_label_dict[img_path] = AU_set

            # print("begin fetch cropped image and bbox {}".format(img_path))
            key_prefix = self.database + "|"
            key = key_prefix+ "/".join((img_path.split("/")[-3], img_path.split("/")[-2],img_path.split("/")[-1]))
            landmark_dict = None
            AU_box_dict = None
            if self.mc_manager is not None and key in self.mc_manager:
                result = self.mc_manager.get(key)
                landmark_dict = result.get("landmark_dict",None)
                AU_box_dict = result.get("AU_box_dict", None)
            parallel_data.append((img_path, landmark_dict, AU_box_dict))
        with Pool(processes=3) as pool:
            parallel_result = pool.starmap_async(parallel_landmark_and_conn_component, parallel_data)
            img_dict = dict()
            for img_path, *_ in parallel_data:
                img_dict[img_path] = cv2.imread(img_path, cv2.IMREAD_COLOR)
            parallel_result.wait()
        # pool.close()
        # pool.join()

        for img_path, AU_box_dict, landmark_dict, box_is_whole_image in parallel_result.get():
            cropped_face = img_dict[img_path]
            rect = None
            if landmark_dict is not None:
                cropped_face, rect = FaceMaskCropper.dlib_face_crop(img_dict[img_path], landmark_dict)
            cropped_face = cv2.resize(cropped_face, config.IMG_SIZE)
            cropped_face = np.transpose(cropped_face, (2, 0, 1))  # put channel first!
            AU_set = img_path_label_dict[img_path]
            key_prefix = self.database + "|"
            key = key_prefix + "/".join((img_path.split("/")[-3], img_path.split("/")[-2], img_path.split("/")[-1]))
            if self.mc_manager is not None:
                save_dict = {"landmark_dict": landmark_dict, "AU_box_dict": AU_box_dict, "crop_rect":rect}
                self.mc_manager.set(key, save_dict)

            AU_couple_gt_label = defaultdict(set)  # key = AU couple, value = AU 用于合并同一个区域的不同AU
            couple_box_dict = DefaultOrderedDict(list)  # key= AU couple

            # mask_path_dict's key AU maybe 3 or -2 or ?5
            if box_is_whole_image:
                for AU in config.AU_SQUEEZE.values():
                    AU_couple_gt_label[self.au_couple_dict[AU]] = AU_set
            else:
                for AU in config.AU_SQUEEZE.values():
                    if AU in AU_set:
                        AU_couple_gt_label[self.au_couple_dict[AU]].add(AU)

            for AU, box_list in sorted(AU_box_dict.items(), key=lambda e: int(e[0])):
                assert AU.isdigit()
                couple_box_dict[self.au_couple_dict[AU]] = box_list # couple_box_dict will contain all AU including not occur on face
            label = []  # one box may have multiple labels. so each entry is 10101110 binary code
            bbox = []
            self.assign_label(couple_box_dict, AU_couple_gt_label, bbox, label)
            assert len(bbox) > 0
            # print("assigned label over")
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            assert bbox.shape[0] == label.shape[0]
            yield cropped_face, bbox, label

    def get_example(self, i):
        if i > len(self):
            raise IndexError("Index too large , i = {}".format(i))

        img_path, AU_set, database_name = self.result_data[i]
        sequence_key = self.extract_sequence_key(img_path)
        fetch_list = []
        if not self.shuffle_T:
            while len(fetch_list) < self.sample_frame:
                fetch_list.clear()
                for fetch_i in range(i-self.previous_frame, i+1):
                    if fetch_i < 0:
                        continue
                    img_path, *_ = self.result_data[fetch_i]
                    if self.extract_sequence_key(img_path) != sequence_key:
                        continue
                    fetch_list.append(fetch_i)
                i += 1
        else:
            fetch_list = [random.randint(0, len(self.result_data)-1) for _ in range(i-self.previous_frame, i+1)]
        if self.train_mode and 0 < self.sample_frame < len(fetch_list):
            choice_frame = np.random.choice(np.arange(len(fetch_list)), size=self.sample_frame, replace=False)
            choice_frame = np.sort(choice_frame)
            fetch_list = [fetch_list[frame] for frame in choice_frame]

        sequence_images = []
        sequence_boxes = []
        sequence_labels = []

        assert len(fetch_list) == self.sample_frame, img_path
        for idx, (cropped_face, bbox, label) in enumerate(self.fetch_func(fetch_list)):
            assert cropped_face is not None
            if bbox.shape[0] != config.BOX_NUM[self.database]:
                print("error! image: {0} box number is {1} != {2}".format(self.result_data[idx][0], bbox.shape[0],
                                                                          config.BOX_NUM[self.database]))
                continue
            sequence_images.append(cropped_face)
            sequence_boxes.append(bbox)
            sequence_labels.append(label)
        sequence_images = np.stack(sequence_images)  # T, C, H, W
        sequence_boxes = np.stack(sequence_boxes)    # T, R, 4
        sequence_labels = np.stack(sequence_labels)  # T, R, 22/12
        # assert sequence_images.shape[0] == self.sample_frame, img_path
        if self.paper_report_label_idx:
            sequence_labels = sequence_labels[:, :, self.paper_report_label_idx]

        return sequence_images, sequence_boxes, sequence_labels