import os
from lru import LRU

import chainer
import numpy as np

from collections import defaultdict
from operator import itemgetter
import config
from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
class AULstmDataset(chainer.dataset.DatasetMixin):

    def __init__(self, directory,  database, num_attrib):
        self.directory = directory

        self.dataset = GlobalDataSet(num_attrib)
        self.database = database
        self.file_name_dict = dict()
        self.lru_cache = LRU(10000)
        for idx, path in enumerate(os.listdir(directory)):
            if path.endswith(".txt"):
                self.file_name_dict[idx] = directory + os.sep + path

    def __len__(self):
        return len(self.file_name_dict)


    def get_example(self, i):
        key = "LSTM_{0}".format(self.file_name_dict[i])
        if key in self.lru_cache:
            x,t = self.lru_cache[key]
            return x,t  # list of T x D and list of T x Y
        sample = self.dataset.load_data(self.file_name_dict[i])
        x_dict = {} # node_id => feature
        t_dict = {} # node_id => AU_bin (ground truth)
        for node in sample.node_list:
            x_dict[node.id] = node.feature
            t_dict[node.id] = node.label

        temporal_link_feature = defaultdict(dict) # key box_id, value = {frame: feature}
        temporal_link_target = defaultdict(dict) # key box_id, value = {frame: gt_label}
        get_box = lambda e_str : e_str.split("_")[1]
        get_frame = lambda e: int(e[0:e.index("_")])
        for edge in sample.edge_list:

            if "temporal" == self.dataset.edge_type_dict.get_key(edge.edge_type):
                a_str = sample.nodeid_line_no_dict.get_key(edge.a)
                b_str = sample.nodeid_line_no_dict.get_key(edge.b)
                box_id_a = get_box(a_str)
                box_id_b = get_box(b_str)
                assert box_id_a == box_id_b
                frame_id_a = int(get_frame(a_str))
                frame_id_b = int(get_frame(b_str))

                node_feature_a = x_dict[edge.a]
                node_feature_b = x_dict[edge.b]

                if frame_id_a not in temporal_link_feature[box_id_a]:
                    temporal_link_feature[box_id_a][frame_id_a] = node_feature_a
                if frame_id_b not in temporal_link_feature[box_id_b]:
                    temporal_link_feature[box_id_b][frame_id_b] = node_feature_b
                if frame_id_a not in temporal_link_target[box_id_a]:
                    temporal_link_target[box_id_a][frame_id_a] = t_dict[edge.a]
                if frame_id_b not in temporal_link_target[box_id_b]:
                    temporal_link_target[box_id_b][frame_id_b] = t_dict[edge.b]

        x = []  # list of T x D
        t = []  # list of T x Y
        assert len(temporal_link_target) == config.BOX_NUM[self.database]
        for box_id, frame_feature_dict in temporal_link_feature.items():
            x.append(np.asarray(list(map(itemgetter(1), sorted(frame_feature_dict.items(), key=lambda e:e[0])))))
            t.append(np.asarray(list(map(itemgetter(1), sorted(temporal_link_target[box_id].items(), key=lambda e: e[0])))))
            # print("frame len:{}".format(len(frame_feature_dict)))
        self.lru_cache[key] = x,t

        return np.stack(x), np.stack(t)  # x shape N x T x D, where N is number of box

