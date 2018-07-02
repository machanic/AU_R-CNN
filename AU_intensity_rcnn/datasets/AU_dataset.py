import os
import random
from collections import defaultdict, OrderedDict

import chainer
import cv2
import numpy as np
import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
from img_toolkit.face_mask_cropper import FaceMaskCropper

# obtain the cropped face image and bounding box and ground truth label for each box
class AUDataset(chainer.dataset.DatasetMixin):

    def __init__(self, img_resolution, database, fold, split_name, split_index, mc_manager, prefix="", pretrained_target=""):
        self.database = database
        self.img_resolution = img_resolution
        self.split_name = split_name
        self.au_couple_dict = get_zip_ROI_AU()
        self.mc_manager = mc_manager
        self.au_couple_child_dict = get_AU_couple_child(self.au_couple_dict)
        self.AU_intensity_label = {}  # subject + "/" + emotion_seq + "/" + frame => ... not implemented
        self.pretrained_target = pretrained_target
        self.dir = config.DATA_PATH[database] # BP4D/DISFA/ BP4D_DISFA

        id_list_file_path = os.path.join(self.dir + "/idx/{0}_fold{1}".format(fold, prefix), "intensity_{0}_{1}.txt".format(split_name, split_index))
        self.result_data = []

        self.video_offset = OrderedDict()
        self.video_count = defaultdict(int)
        print("idfile:{}".format(id_list_file_path))
        with open(id_list_file_path, "r") as file_obj:
            for idx, line in enumerate(file_obj):
                if line.rstrip():
                    line = line.rstrip()
                    img_path, au_set_str, from_img_path, current_database_name = line.split("\t")
                    AU_intensity = np.fromstring(au_set_str, dtype=np.int32, sep=',')
                    from_img_path = img_path if from_img_path == "#" else from_img_path
                    img_path = config.RGB_PATH[current_database_name] + os.path.sep + img_path  # id file 是相对路径
                    from_img_path = config.RGB_PATH[current_database_name] + os.path.sep + from_img_path
                    video_id = "/".join([img_path.split("/")[-3], img_path.split("/")[-2]])
                    if video_id not in self.video_offset:
                        self.video_offset[video_id] = len(self.result_data)
                    self.video_count[video_id] += 1
                    if os.path.exists(img_path):
                        self.result_data.append((img_path, from_img_path, AU_intensity, current_database_name))
        self.result_data.sort(key=lambda entry: (entry[0].split("/")[-3],entry[0].split("/")[-2],
                                                 int(entry[0].split("/")[-1][:entry[0].split("/")[-1].rindex(".")])))
        self._num_examples = len(self.result_data)
        print("read id file done, all examples:{}".format(self._num_examples))

    def __len__(self):
        return self._num_examples

    def assign_label(self, couple_box_dict, current_AU_couple, bbox, label):
        AU_couple_bin = dict()
        for au_couple_tuple, _ in couple_box_dict.items():
            # use connectivity components to seperate polygon
            AU_inside_box = current_AU_couple[au_couple_tuple]  # AU: intensity

            AU_bin = np.zeros(shape=len(config.AU_INTENSITY_DICT), dtype=np.int32)  # 全0表示背景，脸上没有运动
            for AU, intensity in sorted(AU_inside_box.items(), key=lambda e: int(e[0])):
                if AU not in config.AU_SQUEEZE.inv:
                    continue
                idx = config.AU_INTENSITY_DICT.inv[AU]
                np.put(AU_bin, idx, intensity)
            AU_couple_bin[au_couple_tuple] = AU_bin  # for the child
        # 循环两遍，第二遍拿出child_AU_couple
        for au_couple_tuple, box_list in couple_box_dict.items():
            AU_child_bin = np.zeros(shape=len(config.AU_INTENSITY_DICT), dtype=np.int32)
            if au_couple_tuple in self.au_couple_child_dict:
                for au_couple_child in self.au_couple_child_dict[au_couple_tuple]:
                    AU_child_bin = np.maximum(AU_child_bin, AU_couple_bin[au_couple_child])
            AU_bin_tmp = AU_couple_bin[au_couple_tuple]  # 全0表示背景，脸上没有运动
            AU_bin = np.maximum(AU_child_bin, AU_bin_tmp)
            bbox.extend(box_list)
            for _ in box_list:
                label.append(AU_bin)


    def get_example(self, i):
        '''
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        :param i:  the index of the example
        :return: tuple of an image and its all bounding box
        '''
        if i > len(self.result_data):
            raise IndexError("Index too large")
        img_path, from_img_path, AU_intensity, database_name = self.result_data[i]
        if not os.path.exists(img_path):
            raise IndexError("image file_path: {} not exist!".format(img_path))
        try:
            # print("begin fetch cropped image and bbox {}".format(img_path))
            read_img_path = img_path if from_img_path == "#" else from_img_path
            rgb_img_path = config.RGB_PATH[self.database] + os.path.sep + os.path.sep.join(read_img_path.split("/")[-3:])
            key_prefix = self.database +"@{}".format(self.img_resolution) +"|"
            if self.pretrained_target is not None and len(self.pretrained_target) > 0:
                key_prefix = self.pretrained_target+"|"

            cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(read_img_path, rgb_img_path,
                                                                               channel_first=True,
                                                                               mc_manager=self.mc_manager, key_prefix=key_prefix)
        except IndexError:
            print("crop image error:{}".format(img_path))
            face = np.transpose(cv2.resize(cv2.imread(img_path), config.IMG_SIZE), (2, 0, 1))
            whole_bbox = np.tile(np.array([1, 1, config.IMG_SIZE[1] - 1, config.IMG_SIZE[0] - 1], dtype=np.float32),
                                 (config.BOX_NUM[database_name], 1))
            whole_label = np.tile(AU_intensity, (config.BOX_NUM[database_name], 1))
            return face, whole_bbox, whole_label


        current_AU_couple = defaultdict(dict) # key = AU couple, value = {出现的AU: intensity}
        couple_box_dict = OrderedDict()  # key= AU couple

        for idx, intensity in enumerate(AU_intensity):
            AU = str(config.AU_INTENSITY_DICT[idx])
            if intensity > 0:
                try:
                    current_AU_couple[self.au_couple_dict[AU]][AU] = intensity
                except KeyError:
                    print(list(self.au_couple_dict.keys()), AU)
                    raise
        for AU, box_list in sorted(AU_box_dict.items(), key=lambda e:int(e[0])):
            AU = str(AU)
            couple_box_dict[self.au_couple_dict[AU]] = box_list  # 所以这一步会把脸上有的，没有的AU都加上
        label = []  # one box may have multiple labels. so each entry is 10101110 binary code
        bbox = []  # AU = 0背景的box是随机取的
        self.assign_label(couple_box_dict, current_AU_couple, bbox, label)
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        assert bbox.shape[0] == label.shape[0]
        return cropped_face, bbox, label