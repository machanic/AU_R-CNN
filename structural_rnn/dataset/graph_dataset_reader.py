import numpy as np
import config
if config.OPEN_CRF_CONFIG['use_pure_python']:
    from structural_rnn.model.open_crf.pure_python.constant_variable import LabelTypeEnum
else:
    from structural_rnn.model.open_crf.cython.factor_graph import LabelTypeEnum

from bidict import bidict

import json
import os
class MappingDict(object):

    def __init__(self):
        self.mapping_dict = bidict()  # key: str -> id: int
        self.keys = []

    def get_size(self):
        return len(self.keys)

    def __len__(self):
        assert len(self.keys) == len(self.mapping_dict)
        return self.get_size()

    def get_id(self, key:str):
        if key in self.mapping_dict:
            return self.mapping_dict[key]
        id = len(self.keys)
        self.keys.append(key)
        self.mapping_dict[key] = id
        return id

    def get_key(self, value):
        return self.mapping_dict.inv[value]

    def get_keystr_const(self, id:int):
        if id not in self.mapping_dict.inv:
            return -1
        return self.mapping_dict.inv[id]

    def get_id_const(self, key:str):
        if key not in self.mapping_dict:
            return -1
        return self.mapping_dict[key]

    def get_key_with_id(self,id:int):
        if id < 0 or id > len(self.keys):
            return ""
        return self.keys[id]

    def save_mapping_dict(self, file_path:str):
        with open(file_path, "w") as file_obj:
            for i, key in enumerate(self.keys):
                file_obj.write("{0} {1}\n".format(key, i))
            file_obj.flush()

    def load_mapping_dict(self, file_path):
        self.keys.clear()
        self.mapping_dict.clear()
        with open(file_path, "r") as file_obj:
            for line in file_obj:
                if line:
                    line = line.split()  # 我自己加的一句话
                    key, id = line[0], line[1]  # why???
                    self.keys.append(key)
                    self.mapping_dict[key] = id


class DataNode(object):
    def __init__(self,  id:int, label_type:int, label_bin, feature):
        self.id = id
        self.label_type = label_type
        self.label_num = len(label_bin)  # label_bin is np.ndarray
        self.label_bin = label_bin
        self.feature = feature
        self.num_attrib = len(feature)


    @property
    def label(self):  # note that (0,0,0,0,0...,0) will use label=0, thus we +1 here
        nonzero_idx = np.nonzero(self.label_bin)[0]
        assert len(nonzero_idx) <= 1
        if len(nonzero_idx) > 0:
            return nonzero_idx[0] + 1 # random pick, +1 is for the reason 0 also will be one label
        return 0  # all zero became 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class DataEdge(object):
    def __init__(self, id:int,edge_dict:MappingDict, a:int, b:int, edge_type:int):
        self.id = id
        self.a = a
        self.b = b
        self.edge_type = edge_type
        self.edge_dict = edge_dict


class DataSample(object):

    def __init__(self, num_node=None, num_edge=None,  label_dict=None, file_path=None):
        self.file_path = file_path
        self.num_node = num_node   # note that the num_node attribute of FactorGraph is node count + edge count
        self.num_edge = num_edge
        self.node_list = []
        self.edge_list = []
        self.nodeid_line_no_dict = MappingDict()
        self.label_dict = label_dict
        self.label_bin_len = 0


    def clear(self):
        if self.node_list:
            self.node_list.clear()
        if self.edge_list:
            self.edge_list.clear()

    def __del__(self):
        self.clear()
        self.node_list = None
        self.edge_list = None


class GlobalDataSet(object):

    def __init__(self, info_dict_path):
        assert os.path.exists(info_dict_path), info_dict_path
        self.num_edge_type = 0
        self.num_label = 0
        self.label_dict = MappingDict()  # pred_idx <=> true label
        self.edge_type_dict = MappingDict()
        self.num_attrib_type = 0
        self.info_json = None
        self.load_data_info_dict(info_dict_path)  # {"num_label":233, "non_zero_attrib_index":[0,1,4,5,6,...] }
        self.label_bin_len = 0



    def load_data_info_dict(self, info_dict_path):
        with open(info_dict_path, "r") as file_obj:
            self.info_json = json.loads(file_obj.read())
        self.num_attrib_type = self.info_json["num_attrib_type"]

        for pred_idx, true_label_str in self.info_json["label_dict"].items():
            pred_idx = int(pred_idx)
            self.label_dict.mapping_dict[pred_idx] = true_label_str   # pred_idx <=> true label
            self.label_dict.keys.append(pred_idx)

    def load_data(self, path):
        curt_sample = DataSample()
        curt_sample.file_path = path
        curt_sample.label_dict = self.label_dict
        parent_path = os.path.dirname(os.path.dirname(path)) # cd ../
        base_path = os.path.basename(path)
        npy_path = parent_path + os.sep + base_path[:base_path.rindex(".")] + ".npy"
        h_info_array = np.load(npy_path)
        # main_label is label set which continuous occurrence >= 5, we pick all each main_label as one sample(by deepcopy),
        # only if current label_set doesn't have main_label, we pick up the rest label (minor occurence)
        with open(path, "r") as file_obj:
            for line in file_obj:
                tokens = line.split()
                if tokens[0] == "#edge": # read edge type, 一个文件必须先写node，后写#edge，要不然edge哪知道id对应哪一行
                    assert len(tokens) == 4
                    # note that a and b must start from 0! because nodeid start from 0
                    a = curt_sample.nodeid_line_no_dict.get_id_const(tokens[1])  # 如果没有这个node的行号，返回-1
                    b = curt_sample.nodeid_line_no_dict.get_id_const(tokens[2])  # 如果没有这个node，返回-1
                    edge_type = self.edge_type_dict.get_id(tokens[3])  # 比如temporal 对应的id
                    edge_id = curt_sample.nodeid_line_no_dict.get_id("{0}#{1}&{2}".format(tokens[3],tokens[1],tokens[2]))
                    if a == -1 or b == -1:
                        raise TypeError("#edge error! nodeid={0} or nodeid={1} found in path={2}".format(tokens[1], tokens[2], path))
                    curt_edge = DataEdge(edge_id, self.edge_type_dict, a, b, edge_type=edge_type)  # a 和 b是行号，相当于是node_id，原来OpenCRF代码中这个id是var_node的index
                    curt_sample.edge_list.append(curt_edge)
                else:  # read node, 会将num_label也得到
                    node_id = tokens[0]
                    node_labels = tokens[1]
                    if node_id.startswith("?"):
                        label_type = LabelTypeEnum.UNKNOWN_LABEL
                        node_id = node_id[1:]
                    else:
                        label_type = LabelTypeEnum.KNOWN_LABEL
                    node_id = curt_sample.nodeid_line_no_dict.get_id(node_id)   #nodeid convert to line number int，original string nodeid into dict
                    if node_labels.startswith("(") and node_labels.endswith(")"):  #open-crf cannot use bin form label, but can combine as one to use, located in Node constructor
                        label_bin = np.asarray(list(map(int,node_labels[1:-1].split(",") )), dtype=np.int32)

                    if tokens[2].startswith("feature"):
                        feature_idx = int(tokens[2][len("feature_idx:"):])
                        node_features = h_info_array[feature_idx]

                    else:
                        print("Data format wrong! Label must start with features:/np_file:")
                        return
                    curt_node = DataNode(id=node_id, label_type=label_type, label_bin=label_bin,
                                         feature=node_features)

                    assert len(curt_sample.node_list) == node_id
                    curt_sample.node_list.append(curt_node)
                    self.num_label = len(label_bin) + 1  # label length is bin vector length, 0 will also seems be one label
                    self.label_bin_len = len(label_bin)
                    curt_sample.label_bin_len = len(label_bin)
        if len(curt_sample.node_list) > 0:
            curt_sample.num_node = len(curt_sample.node_list)
            curt_sample.num_edge = len(curt_sample.edge_list)

        # graph desc file read done
        self.num_edge_type = self.edge_type_dict.get_size()  # 这个是所有sample数据文件整体的edge种类
        if self.num_edge_type == 0:
            self.num_edge_type = 1

        return curt_sample








