import os
from lru import LRU

import chainer
import numpy as np

from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure


class S_RNNPlusDataset(chainer.dataset.DatasetMixin):

    def __init__(self, directory, attrib_size, global_dataset=None, need_s_rnn=True):  # attrib_size is only for compute
        # self.mc = mc_manager
        self.need_s_rnn = need_s_rnn
        self.attrib_size = attrib_size
        self.directory = directory
        self.dataset = global_dataset
        self.file_name_dict = dict()
        self.crf_pact_structure_dict = LRU(500)
        for idx, path in enumerate(os.listdir(directory)):
            if path.endswith(".txt"):
                self.file_name_dict[idx] = directory + os.sep + path

    def __len__(self):
        return len(self.file_name_dict)


    def get_example(self, i):
        print("getting i:{}".format(i))
        key = "S_RNN_{0}".format(self.file_name_dict[i])
        if key in self.crf_pact_structure_dict:
            x, crf_pact_structure = self.crf_pact_structure_dict[key]
            return x, crf_pact_structure
        sample = self.dataset.load_data(self.file_name_dict[i])
        # assert sample.num_attrib_type == self.attrib_size
        # crf_pact_structure 的num_attrib控制open-crf层的weight个数，因此必须被设置为hidden_size
        crf_pact_structure = CRFPackageStructure(sample, self.dataset, num_attrib=self.attrib_size, need_s_rnn=self.need_s_rnn)

        x = np.zeros(shape=(len(sample.node_list), crf_pact_structure.num_attrib_type), dtype=np.float32)
        for idx, node in enumerate(sample.node_list):
            assert idx == node.id
            x[node.id, :] = node.feature
        self.crf_pact_structure_dict[key] = (x, crf_pact_structure)
        return x, crf_pact_structure

