import numpy as np
import config
if config.OPEN_CRF_CONFIG['use_pure_python']:
    from structural_rnn.model.open_crf.pure_python.constant_variable import LabelTypeEnum
else:
    from structural_rnn.model.open_crf.cython.factor_graph import LabelTypeEnum
import config
from bidict import bidict

from functools import lru_cache
import json

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
    def __init__(self, label_dict:MappingDict, id:int, label_type:int, label_bin, feature):
        self.id = id
        self.label_type = label_type
        self.label_num = len(label_bin)  # label_bin is np.ndarray
        self.label_arr = np.array([non_zero_idx for non_zero_idx in np.nonzero(label_bin)[0]], dtype=np.int32)  # save memory
        self.label_dict = label_dict
        label_str = []
        for label_squeeze_idx in self.label_arr:
            label_str.append(config.AU_SQUEEZE[label_squeeze_idx])
        label_str = ",".join(sorted(label_str))
        self.label_dict.get_id(label_str)

        self.feature = feature
        self.num_attrib = len(feature)


    @property
    def label_bin(self):
        label_bin = np.zeros(self.label_num, dtype=np.int32)
        for non_zero_idx in self.label_arr:
            np.put(label_bin, non_zero_idx, 1)
        return label_bin

    @label_bin.setter
    def label_bin(self, label_bin):
        self.label_arr = np.array([non_zero_idx for non_zero_idx in np.nonzero(label_bin)[0]], dtype=np.uint8)
        self.label_num = len(label_bin)

    @property
    @lru_cache(maxsize=1)
    def label(self):
        label_str = [config.AU_SQUEEZE[label_squeeze_idx] for label_squeeze_idx in self.label_arr]
        label_str = ",".join(sorted(label_str))
        return self.label_dict.get_id_const(label_str)  #  e.g. key = 1,2 (AU1 & AU2), value = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class DataEdge(object):
    def __init__(self, edge_dict:MappingDict, a:int, b:int, edge_type:int):
        self.a = a
        self.b = b
        self.edge_type = edge_type
        self.edge_dict = edge_dict


class DataSample(object):

    def __init__(self, num_node=None, num_edge=None, num_label=None,  label_dict=None, label_bin_len=None):
        self.num_node = num_node   # note that the num_node attribute of FactorGraph is node count + edge count
        self.num_edge = num_edge
        self.node_list = []
        self.edge_list = []
        self.nodeid_line_no_dict = MappingDict()
        self.label_dict = label_dict
        self.label_bin_len = label_bin_len


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
        self.num_edge_type = 0
        self.num_label = 0
        self.label_dict = MappingDict()  # id:int => AU1,AU2 str
        self.edge_type_dict = MappingDict()
        self.num_attrib_type = 0
        self.info_json = None
        self.load_data_info_dict(info_dict_path)  # {"num_label":233, "non_zero_attrib_index":[0,1,4,5,6,...] }


    def load_data_info_dict(self, info_dict_path):
        with open(info_dict_path, "r") as file_obj:
            self.info_json = json.loads(file_obj.read())
        self.num_label = self.info_json["num_label"]
        self.num_attrib_type = len(self.info_json["non_zero_attrib_index"])

    def load_data(self, path):
        curt_sample = DataSample()
        curt_sample.label_dict = self.label_dict

        with open(path, "r") as file_obj:
            for line in file_obj:
                tokens = line.split()
                if tokens[0] == "#edge": # read edge type, 一个文件必须先写node，后写#edge，要不然edge哪知道id对应哪一行
                    assert len(tokens) == 4
                    # note that a and b must start from 0! because nodeid start from 0
                    a = curt_sample.nodeid_line_no_dict.get_id_const(tokens[1])  # 如果没有这个node的行号，返回-1
                    b = curt_sample.nodeid_line_no_dict.get_id_const(tokens[2])  # 如果没有这个node，返回-1
                    edge_type = self.edge_type_dict.get_id(tokens[3])  # 比如temporal 对应的id
                    if a == -1 or b == -1:
                        raise TypeError("#edge error! nodeid={0} or nodeid={1} found in path={2}".format(tokens[1], tokens[2], path))
                    curt_edge = DataEdge(self.edge_type_dict, a, b, edge_type=edge_type)  # a 和 b是行号，相当于是node_id，原来OpenCRF代码中这个id是var_node的index
                    curt_sample.edge_list.append(curt_edge)
                else:  # read node, 会将num_label也得到
                    assert len(tokens) == 3
                    node_id = tokens[0]
                    node_labels = tokens[1]
                    if node_id.startswith("?"):
                        label_type = LabelTypeEnum.UNKNOWN_LABEL
                        node_id = node_id[1:]
                    else:
                        label_type = LabelTypeEnum.KNOWN_LABEL
                    node_id = curt_sample.nodeid_line_no_dict.get_id(node_id)  # nodeid 转成行号int，原始字符串的nodeid放入字典
                    if node_labels.startswith("(") and node_labels.endswith(")"):  # opencrf无法使用这种类型的label，但是
                        label_bin = np.asarray(list(map(int, node_labels[1:-1].split(","))), dtype=np.int32)

                    if tokens[2].startswith("features:"):
                        node_features = np.asarray(list(map(float, tokens[2][len("features:"):].split(","))), dtype=np.float32)
                        node_features = node_features[self.info_json["non_zero_attrib_index"]]

                    else:
                        print("Data format wrong! Label must start with features:/np_file:")
                        return
                    curt_node = DataNode(id=node_id, label_type=label_type, label_dict= self.label_dict, label_bin=label_bin,
                                         feature=node_features)

                    assert len(curt_sample.node_list) == node_id
                    curt_sample.node_list.append(curt_node)
                    curt_sample.label_bin_len = len(label_bin)

        # graph desc file read done
        self.num_edge_type = self.edge_type_dict.get_size()  # 这个是所有sample数据文件整体的edge种类
        if self.num_edge_type == 0:
            self.num_edge_type = 1
        if len(curt_sample.node_list) > 0:
            curt_sample.num_node = len(curt_sample.node_list)
            curt_sample.num_edge = len(curt_sample.edge_list)
        return curt_sample








