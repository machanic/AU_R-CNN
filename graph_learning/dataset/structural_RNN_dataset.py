import os
from lru import LRU

import chainer
import numpy as np

from graph_learning.dataset.crf_pact_structure import CRFPackageStructure


class S_RNNPlusDataset(chainer.dataset.DatasetMixin):

    # note that the target_dict is only for evaluation the test set
    def __init__(self, directory, attrib_size, global_dataset=None, need_s_rnn=True, need_cache_factor_graph=False,
                 need_adjacency_matrix=False, target_dict=None, npy_in_parent_dir=True, need_factor_graph=True):  # attrib_size is only for compute
        # self.mc = mc_manager
        self.need_s_rnn = need_s_rnn
        self.need_factor_graph = need_factor_graph
        self.npy_in_parent_dir = npy_in_parent_dir
        self.need_adj_matrix = need_adjacency_matrix
        self.attrib_size = attrib_size
        self.dataset = global_dataset
        self.file_name_dict = dict()
        self.need_cache = need_cache_factor_graph
        self.crf_pact_structure_dict = LRU(500)
        idx = 0
        if target_dict is not None:
            if len(list(filter(lambda p:p.endswith("txt"), os.listdir(directory)))) == 0:  # test mode , goto sub-folder to classify
                for folder_name in os.listdir(directory):
                    if os.path.isdir(directory + os.sep + folder_name):
                        if folder_name in target_dict:  # only in target_dict file will load
                            for file_name in os.listdir(directory + os.sep + folder_name):
                                self.file_name_dict[idx] = directory + os.sep + folder_name + os.sep +file_name
                                idx+=1
        else:
            for idx, path in enumerate(filter(lambda p:p.endswith("txt"), os.listdir(directory))):
                self.file_name_dict[idx] = directory + os.sep + path

    def __len__(self):
        return len(self.file_name_dict)


    def get_example(self, i):
        print("getting i:{}".format(i))
        key = "S_RNN_{0}".format(self.file_name_dict[i])
        if key in self.crf_pact_structure_dict and self.need_cache:
            x, sample = self.crf_pact_structure_dict[key]

            crf_pact_structure = CRFPackageStructure(sample, self.dataset, num_attrib=self.attrib_size,
                                                     need_s_rnn=self.need_s_rnn, need_adjacency_matrix=self.need_adj_matrix,
                                                     need_factor_graph=self.need_factor_graph)

            return x, crf_pact_structure
        sample = self.dataset.load_data(self.file_name_dict[i], self.npy_in_parent_dir)
        # assert sample.num_attrib_type == self.attrib_size
        # crf_pact_structure 的num_attrib控制open-crf层的weight个数，因此必须被设置为hidden_size
        crf_pact_structure = CRFPackageStructure(sample, self.dataset, num_attrib=self.attrib_size,
                                                 need_s_rnn=self.need_s_rnn, need_adjacency_matrix=self.need_adj_matrix,
                                                 need_factor_graph=self.need_factor_graph)

        x = np.zeros(shape=(len(sample.node_list), self.dataset.num_attrib_type), dtype=np.float32)
        for idx, node in enumerate(sample.node_list):
            assert idx == node.id
            x[node.id, :] = node.feature
        if self.need_cache:
            self.crf_pact_structure_dict[key] = (x, sample)
        return x, crf_pact_structure

