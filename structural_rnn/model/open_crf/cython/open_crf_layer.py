import math
from collections import defaultdict

import chainer
import numpy as np
from chainer import initializers

from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
from structural_rnn.model.open_crf.cython.open_crf import crf_function


class OpenCRFLayer(chainer.Link):

    def __init__(self, node_in_size, weight_len):
        super(OpenCRFLayer, self).__init__()
        self.node_in_size = node_in_size
        with self.init_scope():
            self.W = chainer.Parameter(initializer=initializers.Zero(dtype=np.float32), shape=(weight_len,))

    def __call__(self, xs:chainer.Variable, crf_pact_structures:CRFPackageStructure):  # x指的是一个graph或者一个子图内的node: shape=[n, v] v= 1000是feature数量，n是node个数
        f = 0
        for idx, x in enumerate(xs):  # xs is shape = B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector
            crf_pact = crf_pact_structures[idx]
            f += crf_function(crf_pact, x, self.W, self.node_in_size)  # 每次传给CRF_layer的时候，还要传递给整个时空图数据结构给他
        chainer.reporter.report({
            'loss':f},
            self)
        return f

    def predict(self, x:np.ndarray, crf_pact_structure:CRFPackageStructure):  # x is one video's node feature
        xp = chainer.cuda.get_array_module(x)

        factor_graph = crf_pact_structure.factor_graph
        factor_graph.clear_data_for_sum_product()
        factor_graph.labeled_given = True
        sample = crf_pact_structure.sample
        n = sample.num_node
        m = sample.num_edge
        num_label = crf_pact_structure.num_label
        W_1 = self.W.data[0:num_label * self.node_in_size].reshape(num_label, self.node_in_size)
        WX = np.dot(W_1, x.T).T  # W_1 shape= Y x t ; x.T shape = t x n ==> result matrix = Y x n ==> transpose n x Y
        variable_state_factor = np.exp(WX) # shape = n x Y
        for i in range(n):
            factor_graph.set_variable_state_factor(i, variable_state_factor[i, :])
        factor_graph.belief_propagation(crf_pact_structure.max_bp_iter, self.W.data)
        inf_label = xp.zeros(n, dtype=xp.int32)
        label_prob = xp.zeros(shape=(num_label, n), dtype=xp.float32)
        for i in range(n):
            y_best = -1
            v_best = -999999.0
            v_sum = 0.0
            for y in range(num_label):
                v = factor_graph.var_node[i].state_factor[y]
                for t in range(len(factor_graph.var_node[i].neighbor)):
                    v *= factor_graph.var_node[i].belief[t, y]
                if v > v_best:
                    y_best = y
                    v_best = v
                label_prob[y, i] = v
                v_sum += v
            inf_label[i] = y_best
            for y in range(num_label):
                label_prob[y,i] /= v_sum
        return inf_label
