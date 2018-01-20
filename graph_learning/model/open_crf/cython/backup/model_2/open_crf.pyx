# cython: profile=True
# cython: linetrace=True
import numpy as np
cimport numpy as np
DTYPE_float = np.float32
DTYPE_int = np.int32
DTYPE_obj = np.object
ctypedef np.float32_t DTYPE_float_t
ctypedef np.int32_t DTYPE_int_t
from graph_learning.openblas_toolkit.openblas_configure import num_threads
from libc.math cimport exp, fabs,log,sqrt

import cython
import chainer
from chainer import Function
from chainer import utils
from chainer.utils import type_check
from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
from collections import defaultdict
import chainer.functions as F
import numpy as np
cimport numpy as np
from structural_rnn.model.open_crf.cython.factor_graph cimport FactorGraph

import time


cdef class CRF_CFunction:
    cdef public FactorGraph factor_graph
    cdef public int max_bp_iter, num_feature, num_attrib_parameter, num_edge_feature_each_type, num_attrib_type,num_edge_type
    cdef dict edge_feature_offset
    cdef public object sample
    cdef public int node_in_size
    cdef public int num_label
    def __init__(self, object crf_pact_structure, int node_in_size):
        self.factor_graph = crf_pact_structure.factor_graph
        self.sample = crf_pact_structure.sample
        self.node_in_size = node_in_size
        assert node_in_size == crf_pact_structure.num_attrib_type
        self.max_bp_iter = crf_pact_structure.max_bp_iter
        self.num_feature = crf_pact_structure.num_feature
        self.edge_feature_offset = crf_pact_structure.edge_feature_offset
        self.num_attrib_parameter = crf_pact_structure.num_label * node_in_size
        self.num_label = crf_pact_structure.num_label
        self.num_edge_feature_each_type = crf_pact_structure.num_edge_feature_each_type
        self.num_edge_type = crf_pact_structure.num_edge_type



    cdef int get_attrib_parameter_id(self, int y, int x):
        return y * self.node_in_size + x

    cdef int get_edge_parameter_id(self, int edge_type, int a, int b):
        cdef int key
        key = self.num_label * (a if a < b else b) + (a if a > b else b)
        offset = self.edge_feature_offset[key]
        return self.num_attrib_parameter + edge_type * self.num_edge_feature_each_type + offset

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double log_likelihood(self, np.ndarray[DTYPE_float_t, ndim=2] x,
                                np.ndarray[DTYPE_float_t, ndim=1] W):
        cdef int n,m,i,y,y1,a,b,a1,b1,t, edge_type
        cdef double v,f,Z,h_e,h_v
        cdef int num_attrib_type = self.node_in_size
        assert num_attrib_type == x.shape[1]
        n = self.sample.num_node
        m = self.sample.num_edge

        self.factor_graph.labeled_given = False
        self.factor_graph.clear_data_for_sum_product()
        cdef np.ndarray[DTYPE_float_t, ndim=2] W_1 = W[0:self.num_label*num_attrib_type].reshape(self.num_label, num_attrib_type)
        cdef np.ndarray[DTYPE_float_t, ndim=2] WX = np.dot(W_1, x.T).T # W_1 shape= Y x t ; x.T shape = t x n ==> result matrix = Y x n ==> transpose n x Y
        cdef np.ndarray[DTYPE_float_t, ndim=2] variable_state_factor = np.exp(WX)
        for i in range(n):
            self.factor_graph.set_variable_state_factor(i, variable_state_factor[i,:])

        # for i in range(n):
        #     for y in range(self.factor_graph.num_label):
        #         v = 0.0
        #         for t in self.x_nonzero_attrib[i]:
        #             v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
        #         self.factor_graph.set_variable_state_factor(i, y, exp(v))  #得到紧致表达
        # must compute marginal
        self.factor_graph.belief_propagation(self.max_bp_iter, W)  # only 10 secs at most

        self.factor_graph.calculate_marginal(W)  # 56 secs
        f = 0.0  # f using no penalty form
        Z = 0.0
        #  \sum \lambda_i * f_i

        for i in range(n):
            h_v = 0.0
            y = self.sample.node_list[i].label  #监督信息
            f += WX[i, y]
            Z += np.sum(WX[i,:] * self.factor_graph.var_node[i].marginal) #using Bethe Approximation
            # for t in self.x_nonzero_attrib[i]:
            #     for y1 in range(self.num_label):
            #         Z += W[self.get_attrib_parameter_id(y1, t)] * \
            #             x[i, t] * self.factor_graph.var_node[i].marginal[y1]
            h_v += (-np.dot(self.factor_graph.var_node[i].marginal, np.log(self.factor_graph.var_node[i].marginal)))
            # for a in range(self.num_label):
            #     if fabs(self.factor_graph.var_node[i].marginal[a]) > 1e-10:
            #         h_v += (-self.factor_graph.var_node[i].marginal[a] * log(self.factor_graph.var_node[i].marginal[a]))
            Z -= h_v * (len(self.factor_graph.var_node[i].neighbor)-1)

        # cache for boost speed
        cdef np.ndarray[DTYPE_float_t, ndim=3] W_2 = np.zeros(shape=(self.num_edge_type, self.num_label, self.num_label), dtype=DTYPE_float)
        for edge_type in range(self.num_edge_type):
            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    W_2[edge_type, a1, b1] = W[self.get_edge_parameter_id(edge_type, a1,b1)]
        # print("begin alloc m={0}, num_label:{1}, num_label:{2}".format(m, self.num_label, self.num_label))
        # cdef np.ndarray[DTYPE_float_t, ndim=3] all_factor_marginal = np.zeros(shape=(m, self.num_label, self.num_label),dtype=DTYPE_float)
        # cdef np.ndarray[DTYPE_float_t, ndim=3] all_factor_W_2 = np.zeros(shape=(m, self.num_label, self.num_label),dtype=DTYPE_float)
        # for i in range(m):
        #     h_e = 0.0 # Edge entropy
        #     a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
        #     b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息, 注意这里，用了数字形式的组合label
        #     f += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)]
        #     all_factor_marginal[i, :, :] = self.factor_graph.factor_node[i].marginal
        #     all_factor_W_2[i, :, :] = W_2[self.sample.edge_list[i].edge_type, :, :]
        # Z += np.sum(all_factor_W_2 * all_factor_marginal)
        # h_e += (-np.sum(all_factor_marginal * np.log(all_factor_marginal)))
        # Z += h_e
        #     # Z += np.sum(W_2[self.sample.edge_list[i].edge_type, :, :] * self.factor_graph.factor_node[i].marginal)
        #     # h_e += (-np.sum(self.factor_graph.factor_node[i].marginal * np.log(self.factor_graph.factor_node[i].marginal)))
        #     # for a1 in range(self.num_label):
        #     #     for b1 in range(self.num_label):
        #     #         Z += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1,b1)] * self.factor_graph.factor_node[i].marginal[a1,b1] # 多乘了一项marginal？这才是正确的，上面的似乎少乘了
        #     #         if self.factor_graph.factor_node[i].marginal[a1,b1] > 1e-10:
        #     #             h_e += (- self.factor_graph.factor_node[i].marginal[a1,b1] * log(self.factor_graph.factor_node[i].marginal[a1,b1]))
        #     # Z += h_e
        # f -= Z
        # end = time.time()
        # print("in log-likelihood  end for Z h_e (2) :{}".format(end-start))
        return f


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef calc_gradient(self, np.ndarray[DTYPE_float_t, ndim=2] dx, np.ndarray[DTYPE_float_t, ndim=1] dw,
                        np.ndarray[DTYPE_float_t, ndim=2] x, np.ndarray[DTYPE_float_t, ndim=1] W, bint labeled_given):
        cdef int n, m,i,y,t,a,b,y_gt, a1,b1, index
        cdef int num_attrib_type = self.node_in_size
        assert num_attrib_type == x.shape[1]
        n = self.sample.num_node
        m = self.sample.num_edge

        self.factor_graph.clear_data_for_sum_product()
        self.factor_graph.labeled_given = labeled_given
        cdef np.ndarray[DTYPE_float_t, ndim=2] W_1 = W[0:self.num_label * num_attrib_type].reshape(self.num_label, num_attrib_type)
        cdef np.ndarray[DTYPE_float_t, ndim=2] WX = np.dot(W_1, x.T).T # W_1 shape= Y x t ; x.T shape = t x n ==> result matrix = Y x n ==> transpose n x Y
        cdef np.ndarray[DTYPE_float_t, ndim=2] variable_state_factor = np.exp(WX)
        for i in range(n):
            self.factor_graph.set_variable_state_factor(i, variable_state_factor[i,:])
            # for y in range(self.factor_graph.num_label):
            #     v = 0.0
            #     for t in self.x_nonzero_attrib[i]:
            #         v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
            #     self.factor_graph.set_variable_state_factor(i, y, exp(v))  #得到紧致表达
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        self.factor_graph.calculate_marginal(W)
        # dw shape = num_attrib x num_label + num_edge_type x num_label x num_label
        # dx shape = num_node x num_attrib
        cdef np.ndarray[DTYPE_float_t, ndim=2] dw_1 = dw[0: num_attrib_type* self.num_label].reshape(self.num_label, num_attrib_type)  # change value of dw_1 will also change value of dw
        cdef np.ndarray[DTYPE_float_t, ndim=2] var_P = np.zeros(shape=(n, self.num_label), dtype=DTYPE_float)
        for i in range(n):
            y_gt = self.sample.node_list[i].label  # ground truth supervise information
            dw_1[y_gt, :] += x[i,:] # trick of boost speed
            dx[i, :] += W_1[y_gt, :]  # trick of boost speed
            var_P[i, :] = self.factor_graph.var_node[i].marginal  # trick of boost speed
            # for t in self.x_nonzero_attrib[i]:
            #     dw[self.get_attrib_parameter_id(y_gt, t)] += x[i,t]
            #     dx[i,t] += W[self.get_attrib_parameter_id(y_gt, t)]
            #     for y in range(self.num_label):
            #         dw[self.get_attrib_parameter_id(y, t)] \
            #             -= x[i, t] * self.factor_graph.var_node[i].marginal[y]   # note that y_gt comes from ground truth, y comes from iterator over Y
            #
            #         dx[i,t] -= W[self.get_attrib_parameter_id(y, t)] * \
            #                     self.factor_graph.var_node[i].marginal[y]
        dw_1 -= np.dot(var_P.T, x) # trick of boost speed
        dx -= np.dot(var_P, W_1) # trick of boost speed

        # trick: cache for boost speed
        cdef dict dw_2_reshape = dict()
        for edge_type in range(self.num_edge_type):
            rearrange_index = []
            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    index = self.get_edge_parameter_id(edge_type, a1,b1)
                    rearrange_index.append(index)
            dw_2_reshape[edge_type] = dw[rearrange_index].reshape(self.num_label,self.num_label)
        cdef int gt_edge_type


        for i in range(m):
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息, 注意这里，用了数字形式的label
            dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)] += 1

            gt_edge_type = self.sample.edge_list[i].edge_type
            dw_2_reshape[gt_edge_type] -= self.factor_graph.factor_node[i].marginal
            # below is original (not trick for speed up code)
            # for a1 in range(self.num_label):
            #     for b1 in range(self.num_label):
            #         dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1, b1)] \
            #             -= self.factor_graph.factor_node[i].marginal[a1, b1]







class CRFFunction(Function):
    def __init__(self, crf_pact_structure, int node_in_size):
        self.crf = CRF_CFunction(crf_pact_structure, node_in_size)
        self.crf_pact_structure = crf_pact_structure
        assert crf_pact_structure.num_attrib_type == node_in_size
        self.node_in_size = node_in_size

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            w_type.dtype == np.float32,
            x_type.ndim == 2,
            x_type.shape[0] == self.crf_pact_structure.num_node,
            x_type.shape[1] == self.node_in_size,  # x shape = N x D
            w_type.shape[0] == self.crf.num_feature,)

    def backward(self, tuple inputs,tuple grad_outputs):
        cdef int i
        cdef np.ndarray[DTYPE_float_t, ndim=2] x
        cdef np.ndarray[DTYPE_float_t, ndim=1] W
        cdef double g_norm
        x, W = inputs
        gw, = grad_outputs  # the same shape as W?
        cdef np.ndarray[DTYPE_float_t, ndim=1] dw = np.zeros(shape=(self.crf.num_feature,), dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=2] dx = np.zeros_like(x, dtype=DTYPE_float)
        with num_threads(8):
            self.crf.calc_gradient(dx, dw, x, W, labeled_given=True)
            self.crf.calc_gradient(dx, dw, x, W, labeled_given=False)
        dw *= -1.0
        dx *= -1.0
        # normalize gradient
        g_norm = sqrt(np.sum(dw*dw))
        if g_norm > 1e-8:
            dw /= g_norm

        return gw * dx, gw * dw

    def forward(self, tuple inputs):
        '''
        :param inputs: x shape = (n,1000), W shape=(num_feature, )
        :return:
        '''
        cdef np.ndarray[DTYPE_float_t, ndim=2] x
        cdef np.ndarray[DTYPE_float_t, ndim=1] W
        cdef double f
        x, W = inputs  # x shape=(n, 1000)是一个np.array，是按照node_id的从小到大的顺序排列的，feature_value,具体哪个feature对应哪个node_id
        # 可以从OpenCRFLayer查出来，OpenCRFLayer带一个字典
        # for i in range(num_feature):
        #     f -= (math.pow(W[i], 2) / (2 * math.pow(self.conf.penalty_sigma_square, 2)))  # L2 penalty
        f =0.0
        with num_threads(8):
            f = self.crf.log_likelihood(x, W)
            f *= -1.
        return utils.force_array(f, dtype=W.dtype),

cpdef crf_function( crf_pact_structure:CRFPackageStructure,x: chainer.Variable , W:chainer.Parameter, int node_in_size):
    x = F.copy(x, dst=-1)  # copy to host memory
    return CRFFunction(crf_pact_structure, node_in_size)(x, W)


