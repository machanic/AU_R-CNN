import random
from collections import defaultdict

import chainer
import numpy as np
import config
import os
from itertools import groupby
import math


class NpzFeatureDataset(chainer.dataset.DatasetMixin):

    def __init__(self, directory, database, paper_report_label_idx=None):
        super(NpzFeatureDataset, self).__init__()
        self.database = database
        self.directory = directory
        self.paper_report_label_idx = paper_report_label_idx
        self.file_list = []
        for file_name in os.listdir(directory):
            abs_file_path = directory + os.path.sep + file_name
            self.file_list.append(abs_file_path)

    def __len__(self):
        return len(self.file_list)

    def get_example(self, i):
        file_path = self.file_list[i]
        npz_file = np.load(file_path)
        flow_feature = npz_file['flow_feature']  # shape = N x 2048 it is one AU group box's feature
        rgb_feature = npz_file['rgb_feature']

        flow_scale = 1./ math.ceil(rgb_feature.shape[0] / flow_feature.shape[0])  # 0.1

        label = npz_file['label']  # label is N x 12
        AU_group_id = int(file_path[file_path.rindex("#")+1: file_path.rindex(".")])

        label_trans = label.transpose()  # 12, N
        all_start_end_range = defaultdict(list)
        for AU_idx, column in enumerate(label_trans):
            start_idx = 0
            for label, group in groupby(column):
                if label == 1:
                    all_start_end_range[(start_idx, start_idx + len(list(group)))].append(AU_idx)
                start_idx += len(list(group))

        gt_segments_rgb = np.zeros((config.MAX_SEGMENTS_PER_TIMELINE, 2), dtype=np.float32)  # R, 2
        gt_segments_flow = np.zeros((config.MAX_SEGMENTS_PER_TIMELINE, 2), dtype=np.float32)
        labels = np.zeros(config.MAX_SEGMENTS_PER_TIMELINE, dtype=np.int32)

        for idx, ((start_idx, end_idx), AU_idx_list) in enumerate(all_start_end_range.items()):
            flow_start_idx = start_idx * flow_scale
            flow_end_idx = end_idx * flow_scale
            gt_segments_flow[idx] = np.array([flow_start_idx, flow_end_idx], dtype=np.float32)
            single_label = random.choice(AU_idx_list)
            labels[idx] = single_label # 0 means background
            gt_segments_rgb[idx] = np.array([start_idx, end_idx], dtype=np.float32)

        segment_num = len(all_start_end_range)
        assert segment_num > 0, "file_path: {} not segment found".format(file_path)
        # print("read {}".format(file_path))
        return rgb_feature.transpose(), flow_feature.transpose(), gt_segments_rgb, gt_segments_flow,\
                  np.array([AU_group_id, segment_num],dtype=np.int32), labels