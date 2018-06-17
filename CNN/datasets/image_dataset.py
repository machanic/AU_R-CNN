import cv2
import os
from collections import defaultdict, OrderedDict

import chainer
import numpy as np

import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
from img_toolkit.face_mask_cropper import FaceMaskCropper


# obtain the cropped face image and bounding box and ground truth label for each box
class ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, database, fold, split_name, split_index,mc_manager, train_all_data=False,read_type="rgb",
                 pretrained_target="", img_resolution=config.IMG_SIZE[0]):
        self.database = database
        self.img_resolution = img_resolution
        self.split_name = split_name  # trainval or test
        self.read_type = read_type
        self.au_couple_dict = get_zip_ROI_AU()
        self.au_couple_child_dict = get_AU_couple_child(self.au_couple_dict)
        self.AU_intensity_label = {}  # subject + "/" + emotion_seq + "/" + frame => ... not implemented
        self.dir = config.DATA_PATH[database] # BP4D/DISFA/ BP4D_DISFA
        self.pretrained_target = pretrained_target
        self.mc_manager = mc_manager
        if train_all_data:
            id_list_file_path = os.path.join(self.dir + "/idx/{}_fold".format(fold),
                                             "full_pretrain.txt")
        else:
            id_list_file_path = os.path.join(self.dir + "/idx/{0}_fold".format(fold), "id_{0}_{1}.txt".format(split_name, split_index))
        self.result_data = []

        print("idfile:{}".format(id_list_file_path))
        with open(id_list_file_path, "r") as file_obj:
            for idx, line in enumerate(file_obj):
                if line.rstrip():
                    line = line.rstrip()
                    img_path, au_set_str, from_img_path, current_database_name = line.split("\t")
                    AU_set = set(AU for AU in au_set_str.split(',') if AU in config.AU_ROI)
                    if au_set_str == "0":
                        AU_set = set()
                    from_img_path = img_path if from_img_path == "#" else from_img_path

                    img_path = config.RGB_PATH[current_database_name] + os.path.sep + img_path  # id file 是相对路径
                    from_img_path = config.RGB_PATH[current_database_name] + os.path.sep + from_img_path
                    if os.path.exists(img_path):
                        self.result_data.append((img_path, from_img_path, AU_set, current_database_name))
        self.result_data.sort(key=lambda entry: (entry[0].split("/")[-3],entry[0].split("/")[-2],
                                                 int(entry[0].split("/")[-1][:entry[0].split("/")[-1].rindex(".")])))
        self._num_examples = len(self.result_data)
        print("read id file done, all examples:{}".format(self._num_examples))

    def __len__(self):
        return self._num_examples



    def get_example(self, i):
        '''
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        :param i:  the index of the example
        :return: tuple of an image and its all bounding box
        '''
        if i > len(self.result_data):
            raise IndexError("Index too large")
        img_path, from_img_path, AU_set, database_name = self.result_data[i]

        if not os.path.exists(img_path):
            raise IndexError("image file_path: {} not exist!".format(img_path))
        read_img_path = img_path if from_img_path == "#" else from_img_path
        rgb_img_path = config.RGB_PATH[database_name] + os.path.sep + os.path.sep.join(read_img_path.split("/")[-3:])

        flow_img_path = config.FLOW_PATH[database_name] + os.path.sep + os.path.sep.join(read_img_path.split("/")[-3:])
        key_prefix = self.database + "@{0}".format(self.img_resolution) + "|"
        if self.pretrained_target is not None and len(self.pretrained_target) > 0:
            key_prefix = self.pretrained_target + "|"
        if self.read_type == "rgb":
            read_img_path = rgb_img_path
        elif self.read_type == "flow":
            read_img_path = flow_img_path
        try:
            cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(read_img_path, read_img_path,
                                                                             channel_first=True,
                                                                             mc_manager=self.mc_manager,
                                                                             key_prefix=key_prefix)
        except IndexError:
            print("error in crop face: {}".format(read_img_path))
            cropped_face = cv2.imread(read_img_path, cv2.IMREAD_COLOR)
            cropped_face = cv2.resize(cropped_face, config.IMG_SIZE)
            cropped_face = np.transpose(cropped_face, axes=(2,0,1))
        label =  np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)

        for AU in AU_set:
            if AU in config.AU_SQUEEZE.inv:
                label[config.AU_SQUEEZE.inv[AU]] = 1
        return cropped_face, label
