import random
from collections import defaultdict

import chainer
import numpy as np
import config
import os
from itertools import groupby
from time_axis_rcnn.constants.enum_type import TwoStreamMode


class NpzFeatureDataset(chainer.dataset.DatasetMixin):

    def __init__(self, directory, database, two_stream_mode, T):
        super(NpzFeatureDataset, self).__init__()
        self.database = database
        self.directory = directory
        self.T = T
        self.two_stream_mode = two_stream_mode
        self.file_list = []
        for file_name in sorted(os.listdir(directory), key=lambda f: (f[:f.rindex("#")],
                                                                      int(f[f.rindex("#")+1:f.rindex(".")]))):
            abs_file_path = directory + os.path.sep + file_name
            self.file_list.append(abs_file_path)
        print("loading done, total file: {}".format(len(self)))
    def __len__(self):
        return len(self.file_list)

    def get_example(self, i):
        file_path = self.file_list[i]
        npz_file = np.load(file_path)

        if self.two_stream_mode == TwoStreamMode.rgb:
            feature = npz_file['rgb_feature']
        elif self.two_stream_mode == TwoStreamMode.optical_flow:
            feature = npz_file['flow_feature']
        elif self.two_stream_mode == TwoStreamMode.rgb_flow:
            feature = npz_file["fuse_feature"]


        flow_scale = 1./ self.T  # 0.1

        orig_label = npz_file['label']  # label is N x 12, where N is total frame number
        AU_group_id = int(file_path[file_path.rindex("#")+1: file_path.rindex(".")])

        label_trans = orig_label.transpose()  # 12, N
        all_start_end_range = defaultdict(list)
        for AU_idx, column in enumerate(label_trans):
            start_idx = 0
            for label, group in groupby(column):
                if label == 1:
                    all_start_end_range[(start_idx, start_idx + len(list(group)))].append(AU_idx)
                start_idx += len(list(group))

        gt_segments_rgb = np.zeros((config.MAX_SEGMENTS_PER_TIMELINE, 2), dtype=np.float32)  # R, 2
        gt_segments_flow = np.zeros((config.MAX_SEGMENTS_PER_TIMELINE, 2), dtype=np.float32)
        seg_labels = np.zeros(config.MAX_SEGMENTS_PER_TIMELINE, dtype=np.int32)

        for idx, ((start_idx, end_idx), AU_idx_list) in enumerate(all_start_end_range.items()):
            flow_start_idx = start_idx * flow_scale
            flow_end_idx = end_idx * flow_scale
            gt_segments_flow[idx] = np.array([flow_start_idx, flow_end_idx], dtype=np.float32)
            single_label = random.choice(AU_idx_list)
            seg_labels[idx] = single_label # 0 == fg_class number #1, not the background
            gt_segments_rgb[idx] = np.array([start_idx, end_idx], dtype=np.float32)

        segment_num = len(all_start_end_range)
        assert segment_num > 0, "file_path: {} not segment found".format(file_path)
        # print("read {}".format(file_path))
        return feature.transpose(), gt_segments_rgb, gt_segments_flow,\
                  np.array([AU_group_id, segment_num],dtype=np.int32), seg_labels, orig_label, file_path