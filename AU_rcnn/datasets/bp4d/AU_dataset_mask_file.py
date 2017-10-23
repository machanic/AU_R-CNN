from AU_rcnn.utils import read_image
import chainer
import os
from collections import defaultdict, OrderedDict
import numpy as np
from config import ROOT_PATH
import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU
import cv2
from AU_rcnn.links.model.faster_rcnn.utils.proposal_multi_label import ProposalMultiLabel
import glob

class AUDataset(chainer.dataset.DatasetMixin):


    def fetch_orig_BP4D_mask_dict(self, mask_dir):
        orig_AU_mask_path_dict = dict()
        for root, dirs, files in os.walk(mask_dir):
            for file in files:
                if not file.endswith(".png"):
                    continue
                abs_mask_path = root + os.sep + file
                subject_name = abs_mask_path.split(os.sep)[-3]
                video_seq_name = abs_mask_path.split(os.sep)[-2]
                frame = abs_mask_path.split(os.sep)[-1]
                frame = frame[:frame.index("_")]
                AU_ls = file[file.rindex("_") + 1:file.rindex(".")].split(",")
                for _AU in AU_ls:
                    # print(subject_name+"/"+video_seq_name+"/"+frame+"/"+_AU)
                    orig_AU_mask_path_dict[subject_name+"/"+video_seq_name+"/"+frame+"/"+_AU] = abs_mask_path
        return orig_AU_mask_path_dict

    def get_mask_path_dict(self, orig_img_path, AU_set, orig_AU_mask_path_dict): # 还需要装一些背景的AU的mask
        subject_name = orig_img_path.split(os.sep)[-3]
        video_seq_name = orig_img_path.split(os.sep)[-2]
        frame = orig_img_path.split(os.sep)[-1]
        frame = frame[:frame.rindex(".")]
        mask_path_dict = dict()
        non_prefix_AU_set = set()
        for AU in AU_set:
            _AU = AU if not AU.startswith("?") else AU[1:]
            non_prefix_AU_set.add(_AU)

        rest_AU = set(config.AU_ROI.keys()) - non_prefix_AU_set  # AU_set 可能包含?开头
        already_mask_path = set()

        for AU in AU_set:
            if AU != "0":
                _AU = AU if not AU.startswith("?") else AU[1:]
                # au_couple = self.au_couple_dict[_AU]
                # FIXME 因为SQUEEZE会造成AU的个数改变，组合也就变了，但是mask是以前生成的数据，不得以而为之的办法，calculate mask on the fly更好的办法
                # mask_path = "{0}/{1}/{2}/{3}_AU_{4}.png".format(mask_dir, subject_name, video_seq_name,
                #                                                 frame, ",".join(au_couple))
                mask_path = orig_AU_mask_path_dict[subject_name+"/"+video_seq_name+"/"+frame+"/"+_AU]

                if os.path.exists(mask_path):
                    mask_path_dict[AU] = mask_path
                    already_mask_path.add(mask_path)
        for AU in rest_AU:
            # au_couple = self.au_couple_dict[AU]
            # FIXME BUG
            # mask_path = "{0}/{1}/{2}/{3}_AU_{4}.png".format(mask_dir, subject_name, video_seq_name,
            #                                                 frame, ",".join(au_couple))
            mask_path = orig_AU_mask_path_dict[subject_name+"/"+video_seq_name+"/"+frame+"/"+AU]

            if os.path.exists(mask_path) and (not mask_path in already_mask_path):
                mask_path_dict["-{}".format(AU)] = mask_path
                already_mask_path.add(mask_path)

        return mask_path_dict


    def __init__(self, dir, split_name="trainval", split_index=1):
        self.dir = dir
        self.split_name = split_name
        self.au_couple_dict = get_zip_ROI_AU()
        self.AU_mask_dir = self.dir + "/BP4D_AUmask/"
        self._video_seq = defaultdict(dict)  # subject+"/"+emotion_seq => {frame_no : absolute_path_list}
        self.AU_occur_label = {}  # subject + "/" + emotion_seq + "/" + frame => {"1":1, "2":0} which is "AU": 1/0
        self.AU_intensity_label = {}  # subject + "/" + emotion_seq + "/" + frame => ... not implemented
        id_list_file_path = os.path.join(self.dir + "/idx/", "id_{0}_{1}.txt".format(split_name, split_index))
        self.result_data = []
        orig_AU_mask_path_dict = self.fetch_orig_BP4D_mask_dict(self.AU_mask_dir)

        with open(id_list_file_path, "r") as file_obj:
            for idx, line in enumerate(file_obj):
                if line.rstrip():
                    line = line.rstrip()
                    img_path, au_set_str, from_img_path, database_name = line.split("\t")
                    from_img_path = img_path if from_img_path == "#" else from_img_path

                    img_path = ROOT_PATH + os.sep + img_path  # id file 是相对路径
                    from_img_path = ROOT_PATH + os.sep + from_img_path

                    mask_path_dict = self.get_mask_path_dict(from_img_path, set(au_set_str.split(",")),
                                                             orig_AU_mask_path_dict)
                    if len(mask_path_dict) > 0:
                        self.result_data.append((img_path, from_img_path, database_name,
                                                 mask_path_dict))
        self.proposal = ProposalMultiLabel()
        self._num_examples = len(self.result_data)

    def __len__(self):
        return self._num_examples


    def assign_label(self, couple_mask_dict, current_AU_couple, bbox, label):
        for au_couple_tuple, mask in couple_mask_dict.items():
            # use connectivity components to seperate polygon
            AU_inside_box_set = current_AU_couple[au_couple_tuple]
            AU_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.byte)  # 全0表示背景，脸上没有运动
            for AU in AU_inside_box_set:  # AU_inside_box_set may has -3 or ?3
                if (not AU.startswith("?")) and (not AU.startswith("-")):
                    AU_squeeze = config.AU_SQUEEZE.inv[AU]  # AU_squeeze type = int
                    np.put(AU_bin, AU_squeeze, 1)
                elif AU.startswith("?"):
                    AU_squeeze = config.AU_SQUEEZE.inv[AU[1:]]
                    np.put(AU_bin, AU_squeeze, -1)  # ignore label

            connect_arr = cv2.connectedComponents(mask[0], connectivity=4, ltype=cv2.CV_32S) # mask shape = 1 x H x W
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]
            # convert mask polygon to rectangle
            for component_label in range(1, component_num):

                row_col = list(zip(*np.where(label_matrix == component_label)))
                row_col = np.array(row_col)
                y_min_index = np.argmin(row_col[:, 0])
                y_min = row_col[y_min_index, 0]
                x_min_index = np.argmin(row_col[:, 1])
                x_min = row_col[x_min_index, 1]
                y_max_index = np.argmax(row_col[:, 0])
                y_max = row_col[y_max_index, 0]
                x_max_index = np.argmax(row_col[:, 1])
                x_max = row_col[x_max_index, 1]
                # same region may be shared by different AU, we must deal with it
                coordinates = (y_min, x_min, y_max, x_max)


                if y_min == y_max and x_min == x_max:  # 尖角处会产生孤立的单个点，会不会有一个mask只有尖角？
                    # print(("single point mask: img:{0} mask:{1}".format(self._images[i], mask_path)))
                    continue
                if coordinates not in bbox:
                    bbox.append(coordinates)
                    label.append(AU_bin)
            del label_matrix
            del mask

    def get_example(self, i):
        '''
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        :param i:  the index of the example
        :return: tuple of an image and its all bounding box
        '''
        if i > len(self):
            raise IndexError("Index too large")
        img_path, from_img_path, database_name, mask_path_dict = self.result_data[i]
        img = read_image(img_path, color=True)
        flip = "flip" in img_path
        label = []  # one box may have multiple labels. so each entry is 10101110 binary code
        bbox = [] # AU = 0背景的box是随机取的

        current_AU_couple = defaultdict(set) # key = AU couple, value = AU 用于合并同一个区域的不同AU
        couple_mask_dict = defaultdict(list) # key= AU couple
        # mask_path_dict's key AU maybe 3 or -2 or ?5
        for AU, mask_path in mask_path_dict.items():
            _AU = AU
            if AU.startswith("?") or AU.startswith("-"):
                _AU = AU[1:]
            current_AU_couple[self.au_couple_dict[_AU]].add(AU)  # value list may contain ?2 or -1, 所以这一步会把脸上有的，没有的AU都加上

            mask = read_image(mask_path, dtype=np.uint8, color=False)
            if flip:
                mask = np.fliplr(mask)
            couple_mask_dict[self.au_couple_dict[_AU]] = mask

        self.assign_label(couple_mask_dict, current_AU_couple, bbox, label)

        if len(bbox) == 0:
            raise IndexError()
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        bbox, label = self.proposal(bbox, label)
        assert bbox.shape[0] == label.shape[0]
        return img, bbox, label  # 注意最后返回的label是压缩过的label长度，并且label是01000100这种每个box一个二进制的ndarray的情况