import random
from collections import defaultdict

import chainer
import numpy as np
import config
import os
from itertools import groupby


class SimpleFeatureDataset(chainer.dataset.DatasetMixin):

    def __init__(self, directory, database, paper_report_label_idx=None):
        super(SimpleFeatureDataset, self).__init__()
        self.database = database
        self.directory = directory
        self.paper_report_label_idx = paper_report_label_idx
        self.file_list = []
        for file_name in os.listdir(directory):
            abs_file_path = directory + os.path + file_name
            self.file_list.append(abs_file_path)

    def __len__(self):
        return len(self.file_list)

    def get_example(self, i):
        file_path = self.file_list[i]
        npz_file = np.load(file_path)
        feature = npz_file['feature']  # shape = N x 2048 it is one AU group box's feature
        label = npz_file['label']  # label is N x 12
        AU_group_id = npz_file["AU_group_id"]  # AU_group_id
        label_trans = label.transpose()  # 12, N
        all_start_end_range = defaultdict(list)
        for AU_idx, column in enumerate(label_trans):
            start_idx = 0
            for label, group in groupby(column):
                if label == 1:
                    all_start_end_range[(start_idx, start_idx + len(list(group)))].append(AU_idx)
                start_idx += len(list(group))

        gt_segments = np.zeros((config.MAX_SEGMENTS_PER_TIMELINE, 2), dtype=np.float32)  # R, 2
        labels = np.zeros(config.MAX_SEGMENTS_PER_TIMELINE, dtype=np.int32)
        for idx, ((start_idx, end_idx), AU_idx_list) in enumerate(all_start_end_range.items()):
            single_label = random.choice(AU_idx_list)
            labels[idx] = single_label
            gt_segments[idx] = np.array([start_idx, end_idx], dtype=np.int32)


        segment_num = len(all_start_end_range)
        return feature, np.array([AU_group_id, segment_num],dtype=np.int32), gt_segments, labels