import math
from collections import defaultdict

import chainer
import chainer.functions as F
import numpy as np
from chainer import Function
from chainer import initializers
from chainer import utils
from chainer.utils import type_check

from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
from structural_rnn.model.open_crf.cython.factor_graph import LabelTypeEnum


class CRFFunction(Function):

    def __init__(self, crf_pact_structure:CRFPackageStructure, node_in_size:int):
        self.factor_graph = crf_pact_structure.factor_graph
        self.sample = crf_pact_structure.sample
        self.max_bp_iter = crf_pact_structure.max_bp_iter
        self.num_feature = crf_pact_structure.num_feature
        self.edge_feature_offset = crf_pact_structure.edge_feature_offset
        self.num_attrib_parameter = crf_pact_structure.num_label * node_in_size
        self.num_edge_feature_each_type = crf_pact_structure.num_edge_feature_each_type
        self.x_nonzero_attrib = defaultdict(list)
        self.node_in_size = node_in_size
        self.num_label = crf_pact_structure.num_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            w_type.dtype == np.float32,
            x_type.ndim == 2,
            x_type.shape[0] == self.sample.num_node,
            # x_type.shape[1] == self.node_in_size,
            w_type.shape[0] == self.num_feature,
        )

    def get_attrib_parameter_id(self, y:int, x:int):
        return y * self.node_in_size + x

    def get_edge_parameter_id(self, edge_type:int, a:int, b:int):
        key = self.num_label * (a if a < b else b) + (a if a > b else b)
        offset = self.edge_feature_offset[key]
        return self.num_attrib_parameter + edge_type * self.num_edge_feature_each_type + offset

    def log_likelihood(self, x, W):

        n = self.sample.num_node
        m = self.sample.num_edge

        self.factor_graph.labeled_given = False
        self.factor_graph.clear_data_for_sum_product()
        for i in range(n):
            p_lambda = W
            for y in range(self.factor_graph.num_label):
                v = 1.0
                for t in self.x_nonzero_attrib[i]:
                    v *= math.exp(p_lambda[t] * x[i, t]) # 将self.sample.node_list[i].feature[t]改为x[i,t]
                self.factor_graph.set_variable_state_factor(i, y, v)  #得到紧致表达
                p_lambda = p_lambda[self.node_in_size:]
        # 需要算出marginal
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        self.factor_graph.calculate_marginal(W)
        f = 0.0  # f using no penalty form
        Z = 0.0
        #  \sum \lambda_i * f_i
        for i in range(n):
            y = self.sample.node_list[i].label  #监督信息
            for t in self.x_nonzero_attrib[i]:
                f += W[self.get_attrib_parameter_id(y, t)] * x[i, t]
        for i in range(m):
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息
            f += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)]
        # calc log-likelihood, using Bethe Approximation
        for i in range(n):
            for y in range(self.num_label):
                for t in self.x_nonzero_attrib[i]:
                    Z += W[self.get_attrib_parameter_id(y, t)] * \
                         x[i, t] * self.factor_graph.var_node[i].marginal[y]
        for i in range(m):
            for a in range(self.num_label):
                for b in range(self.num_label):
                    Z += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a,b)] * self.factor_graph.factor_node[i].marginal[a][b] # 多乘了一项marginal？这才是正确的，上面的似乎少乘了
        # Edge entropy
        for i in range(m):
            h_e = 0.0
            for a in range(self.num_label):
                for b in range(self.num_label):
                    if self.factor_graph.factor_node[i].marginal[a][b] > 1e-10:
                        h_e += - self.factor_graph.factor_node[i].marginal[a][b] * math.log(self.factor_graph.factor_node[i].marginal[a][b], math.e)
            Z += h_e
        # Node entropy
        for i in range(n):
            h_v = 0.0
            for a in range(self.num_label):
                if math.fabs(self.factor_graph.var_node[i].marginal[a]) > 1e-10:
                    h_v += (-self.factor_graph.var_node[i].marginal[a] * math.log(self.factor_graph.var_node[i].marginal[a], math.e))
            Z -= h_v * (len(self.factor_graph.var_node[i].neighbor)-1)
        f -= Z
        return f

    def forward_cpu(self, inputs):
        '''
        :param inputs: x shape = (n,1000), W shape=(num_feature, )
        :return:
        '''
        x, W = inputs  # x shape=(n, 1000)是一个np.array，是按照node_id的从小到大的顺序排列的，feature_value,具体哪个feature对应哪个node_id
        # 可以从OpenCRFLayer查出来，OpenCRFLayer带一个字典
        # for i in range(num_feature):
        #     f -= (math.pow(W[i], 2) / (2 * math.pow(self.conf.penalty_sigma_square, 2)))  # L2 penalty
        nonzero_tuple = np.nonzero(x)
        for i,j in zip(*nonzero_tuple):
            self.x_nonzero_attrib[i].append(j)
        f = 0
        # f = self.log_likelihood(x, W)  #FIXME 注释掉加快速度
        f *= -1.
        return utils.force_array(f, dtype=W.dtype),

    def calc_gradient(self, dx, dw, x, W, labeled_given:bool):
        n = self.sample.num_node
        m = self.sample.num_edge
        #
        self.factor_graph.labeled_given = labeled_given
        self.factor_graph.clear_data_for_sum_product()
        for i in range(n):
            p_lambda = W
            for y in range(self.num_label):
                if (self.sample.node_list[i].label_type == LabelTypeEnum.KNOWN_LABEL \
                            and y != self.sample.node_list[i].label): # 监督信息
                    self.factor_graph.set_variable_state_factor(i, y, 0)
                else:
                    v = 1.0
                    for t in self.x_nonzero_attrib[i]:
                        v *= math.exp(p_lambda[t] * x[i, t])
                    self.factor_graph.set_variable_state_factor(i, y, v)
                p_lambda = p_lambda[self.node_in_size:]
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        self.factor_graph.calculate_marginal(W)
        # dw shape = num_attrib x num_label + num_edge_type x num_label x num_label
        # dx shape = num_node x num_attrib
        for i in range(n):
            y_gt = self.sample.node_list[i].label  # ground truth supervise information
            for t in self.x_nonzero_attrib[i]:
                dw[self.get_attrib_parameter_id(y_gt, t)] += x[i, t]
                dx[i, t] += W[self.get_attrib_parameter_id(y_gt, t)]
                for y in range(self.num_label):
                    dw[self.get_attrib_parameter_id(y, t)] \
                        -= x[i, t] * self.factor_graph.var_node[i].marginal[y]
                    dx[i, t] -= W[self.get_attrib_parameter_id(y, t)] * \
                                self.factor_graph.var_node[i].marginal[y]
        for i in range(m):
            a = self.sample.node_list[self.sample.edge_list[i].a].label  # supervise information
            b = self.sample.node_list[self.sample.edge_list[i].b].label  # supervise information
            dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)] += 1

            for a in range(self.num_label):
                for b in range(self.num_label):
                    dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)] -= \
                        self.factor_graph.factor_node[i].marginal[a][b]





    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs
        gw, = grad_outputs  # the same shape as W?
        dw = np.zeros(shape=(self.num_feature, ), dtype=np.float32)
        dx = np.zeros_like(x, dtype=np.float32)
        self.calc_gradient(dx, dw, x, W,labeled_given=False)
        self.calc_gradient(dx, dw, x, W, labeled_given=True)
        dw *= -1.
        dx *= -1.
        # normalize gradient
        g_norm = math.sqrt(np.sum(dw*dw))
        if g_norm > 1e-8:
            dw /= g_norm

        return gw * dx, gw * dw


def crf_function(crf_pact_structure:CRFPackageStructure, x:chainer.Variable, W:chainer.Parameter, node_size:int):
    x = F.copy(x, dst=-1)  # copy to host memory
    return CRFFunction(crf_pact_structure, node_size)(x, W)


class OpenCRFLayer(chainer.Link):

    def __init__(self,  node_in_size, weight_len):
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
        nonzero_tuple = xp.nonzero(x)
        x_nonzero_attrib = defaultdict(list)
        for i, j in zip(*nonzero_tuple):
            x_nonzero_attrib[i].append(j)
        factor_graph = crf_pact_structure.factor_graph
        factor_graph.clear_data_for_sum_product()
        factor_graph.labeled_given = False
        sample = crf_pact_structure.sample
        n = sample.num_node
        m = sample.num_edge
        for i in range(n):
            p_lambda = self.W.data
            for y in range(crf_pact_structure.num_label):
                v = 1.0
                for t in x_nonzero_attrib[i]:
                    v *= math.exp(p_lambda[t] * x[i, t])
                factor_graph.set_variable_state_factor(i, y, v)
                p_lambda = p_lambda[self.node_in_size:]
        factor_graph.belief_propagation(crf_pact_structure.max_bp_iter, self.W.data)
        inf_label = xp.zeros(n, dtype=xp.int32)
        label_prob = xp.zeros(shape=(crf_pact_structure.num_label, n), dtype=xp.float32)
        for i in range(n):
            y_best = -1
            v_best = -999999.0
            v_sum = 0.0
            for y in range(crf_pact_structure.num_label):
                v = factor_graph.var_node[i].state_factor[y]
                for t in range(len(factor_graph.var_node[i].neighbor)):
                    v *= factor_graph.var_node[i].belief[t][y]
                if v > v_best:
                    y_best = y
                    v_best = v
                label_prob[y][i] = v
                v_sum += v
            inf_label[i] = y_best
            for y in range(crf_pact_structure.num_label):
                label_prob[y][i] /= v_sum
        return inf_label
