import chainer
import numpy as np
import config
import os
from collections import defaultdict
from collections_toolkit.ordered_default_dict import DefaultOrderedDict
class SimpleFeatureDataset(chainer.dataset.DatasetMixin):

    def __init__(self, directory, database, each_file_pic_num, previous_frame,
                 sample_pic_count, paper_report_label_idx=None, train_mode=True):
        super(SimpleFeatureDataset, self).__init__()
        self.database = database
        self.sample_pic_count = sample_pic_count
        self.previous_frame = previous_frame
        self.directory = directory
        self.each_file_pic_num = each_file_pic_num
        file_count, sequence_name_list = self.file_count()
        self.paper_report_label_idx = paper_report_label_idx

        self.sequence_name_list = sequence_name_list
        self.fetch_idx_to_file_position = {}
        self.train_mode = train_mode
        fetch_idx = 0
        seq_file_start = set()
        for file_name, pic_count in sorted(file_count.items(), key=lambda e: (e[0][0: e[0].index("@")],
                                           int(e[0][e[0].rindex("@")+1:e[0].rindex(".")]))):
            seq_name = file_name[:file_name.rindex("@")]
            start = 3  # the first file in seq_name will be start from 3-rd frame
            if seq_name in seq_file_start:
                start = 0
            seq_file_start.add(seq_name)
            for i in range(start, pic_count):  # first 2 frame will not input to LSTM
                self.fetch_idx_to_file_position[fetch_idx] = (self.directory + os.sep + file_name, i)
                fetch_idx += 1
        self.total_count = fetch_idx

    def file_count(self):
        file_count = dict()
        sequence_name_list = list()
        for file_name in sorted(os.listdir(self.directory), key=lambda e:(e.split("@")[0],
                                                                          int(e.split("@")[1][:e.split("@")[1].rindex(".")]))):

            npz_file = np.load(self.directory + os.sep + file_name)
            bbox = npz_file["bbox"]
            file_count[file_name] = bbox.shape[0] // config.BOX_NUM[self.database]
            for _ in range(file_count[file_name]):
                sequence_name_list.append(file_name[:file_name.rindex("@")])
        return file_count, sequence_name_list


    def __len__(self):
        return self.total_count

    def get_example(self, i):
        file_path, pic_offset = self.fetch_idx_to_file_position[i]
        previous_pic_offset = pic_offset - self.previous_frame
        rest_fetch_count = previous_pic_offset

        all_feature = []
        all_boxes = []
        all_labels = []
        seq_name = file_path[file_path.rindex(os.sep) + 1: file_path.rindex("@")]
        seq_index = int(file_path[file_path.rindex("@") + 1: file_path.rindex(".")])
        if seq_index > 1:
            if previous_pic_offset < 0:
                previous_seq_index = seq_index - 1
                previous_file_path = "{0}/{1}@{2}.npz".format(os.path.dirname(file_path), seq_name, previous_seq_index)
                while previous_seq_index > 1 and not os.path.exists(previous_file_path):
                    previous_seq_index -= 1
                    previous_file_path = "{0}/{1}@{2}.npz".format(os.path.dirname(file_path), seq_name,
                                                                  previous_seq_index)

                previous_npz_file = np.load(previous_file_path)
                previous_feature = previous_npz_file["feature"].reshape(-1, config.BOX_NUM[self.database], 2048)
                previous_bbox = previous_npz_file["bbox"].reshape(-1, config.BOX_NUM[self.database], 4)
                previous_label = previous_npz_file["label"].reshape(-1, config.BOX_NUM[self.database], len(config.AU_SQUEEZE))
                all_feature.extend(previous_feature[rest_fetch_count:])
                all_boxes.extend(previous_bbox[rest_fetch_count:])
                all_labels.extend(previous_label[rest_fetch_count:])
                previous_pic_offset = 0
        else:  # seq_index <= 0
            if previous_pic_offset < 0:
                previous_pic_offset = 0
        npz_file = np.load(file_path)

        all_feature.extend(npz_file["feature"].reshape(-1, config.BOX_NUM[self.database], 2048)
                           [previous_pic_offset: pic_offset + 1])
        all_boxes.extend(npz_file["bbox"].reshape(-1, config.BOX_NUM[self.database], 4)
                         [previous_pic_offset: pic_offset + 1])
        all_labels.extend(npz_file["label"].reshape(-1, config.BOX_NUM[self.database], len(config.AU_SQUEEZE))
                          [previous_pic_offset: pic_offset + 1])
        feature = np.stack(all_feature)
        boxes = np.stack(all_boxes)
        labels = np.stack(all_labels)
        if self.paper_report_label_idx:
            labels = labels[:, :, self.paper_report_label_idx]
        if self.train_mode:
            choice_frame = np.random.choice(np.arange(boxes.shape[0]), size=self.sample_pic_count, replace=True)
            choice_frame = np.sort(choice_frame)
            feature = feature[choice_frame, ...]
            boxes = boxes[choice_frame, ...]
            labels = labels[choice_frame, ...]
        return feature, boxes, labels  # return shape of (T, box_num_frame, D)