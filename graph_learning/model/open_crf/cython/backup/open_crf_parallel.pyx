

import numpy as np
from joblib import Parallel, delayed
DTYPE_float = np.float32
DTYPE_int = np.int32
DTYPE_obj = np.object
ctypedef np.float32_t DTYPE_float_t
ctypedef np.int32_t DTYPE_int_t


cdef extern from "math.h":
    double exp(double x)
    double log(double x)
    double fabs(double x)
    double sqrt (double x)


import chainer
from chainer import Function
from chainer import utils
from chainer.utils import type_check
from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
import chainer.functions as F
import numpy as np
cimport numpy as np
from structural_rnn.model.open_crf.cython.factor_graph cimport FactorGraph
import copy


cdef class CRF_CFunction:
    cdef public FactorGraph factor_graph
    cdef public int max_bp_iter, num_feature, num_attrib_parameter, num_edge_feature_each_type, num_attrib_type
    cdef dict edge_feature_offset
    cdef public object sample
    cdef public int node_in_size
    cdef public dict marginal_dict
    def __init__(self, object crf_pact_structure, int node_in_size):
        self.factor_graph = crf_pact_structure.factor_graph
        self.sample = crf_pact_structure.sample
        self.node_in_size = node_in_size
        self.max_bp_iter = crf_pact_structure.max_bp_iter
        self.num_feature = crf_pact_structure.num_feature
        self.edge_feature_offset = crf_pact_structure.edge_feature_offset
        self.num_attrib_parameter = crf_pact_structure.num_label * node_in_size

        self.num_edge_feature_each_type = crf_pact_structure.num_edge_feature_each_type
        self.marginal_dict = dict()

    cpdef parallel_belief_propagation(self, FactorGraph factor_graph,
                                               np.ndarray x_asarray, np.ndarray W_asarray,
                                            bint labeled_given, int node_in_size, int max_bp_iter):
        print("parallel_belief_propagation begin")
        cdef np.ndarray[DTYPE_float_t, ndim=2] x = x_asarray
        cdef np.ndarray[DTYPE_float_t, ndim=1] W = W_asarray
        factor_graph.labeled_given = labeled_given
        factor_graph.clear_data_for_sum_product()
        cdef int n = factor_graph.n
        cdef int num_attrib_type = node_in_size
        cdef int m = factor_graph.m
        cdef int i,t,y
        cdef double v
        for i in range(n):
            for y in range(factor_graph.num_label):
                v = 0.0
                for t in np.nonzero(x[i,:])[0]:
                    v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
                factor_graph.set_variable_state_factor(i, y, exp(v))  #得到紧致表达
        factor_graph.belief_propagation(max_bp_iter, W)
        factor_graph.calculate_marginal(W)
        cdef np.ndarray[DTYPE_float_t, ndim=2] var_node_marginal = np.zeros((n, factor_graph.num_label), dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=3] factor_node_marginal = np.zeros((m, factor_graph.num_label, factor_graph.num_label), dtype=DTYPE_float)
        for i in range(n):
            var_node_marginal[i,:] = factor_graph.var_node[i].marginal
        for i in range(m):
            factor_node_marginal[i, :, :] = factor_graph.factor_node[i].marginal
        print("parallel_belief_propagation done!")
        return var_node_marginal, factor_node_marginal



    cdef int get_attrib_parameter_id(self, int y, int x):
        return y * self.node_in_size + x

    cdef int get_edge_parameter_id(self, int edge_type, int a, int b):
        cdef int key
        key = self.sample.num_label * (a if a < b else b) + (a if a > b else b)
        offset = self.edge_feature_offset[key]
        return self.num_attrib_parameter + edge_type * self.num_edge_feature_each_type + offset

    def parallel_bp(self, x, W):
        cdef np.ndarray[DTYPE_float_t, ndim=2] var_node_marginal
        cdef np.ndarray[DTYPE_float_t, ndim=3] factor_node_marginal
        # print("in parallel_bp, self.factor_graph is var_neighbor_len:{0} factor_neighbor_len:{1}".format(len(self.factor_graph.var_node_neighbors), len(self.factor_graph.factor_node_neighbors)))
        cdef FactorGraph copy_factor_graph = copy.deepcopy(self.factor_graph)  # cython does not support deepcopy
        print("deep copy over")
        # print("in parallel_bp, copy_factor_graph is var_neighbor_len:{0} factor_neighbor_len:{1}".format(len(copy_factor_graph.var_node_neighbors), len(copy_factor_graph.factor_node_neighbors)))

        result = Parallel(n_jobs=3)(delayed(self.parallel_belief_propagation)(copy.deepcopy(self.factor_graph),x,W,
                                                                              True if i==1 else False,
                                                                              self.node_in_size, self.max_bp_iter) for i in range(3))
        ret = dict()
        ret["log_likelihood"] = result[0]  # var_node_marginal, factor_node_marginal
        ret["labeled_given"] = result[1]
        ret["labeled_nogiven"] = result[2]
        self.marginal_dict = ret

    cpdef double log_likelihood(self, np.ndarray[DTYPE_float_t, ndim=2] x,
                                np.ndarray[DTYPE_float_t, ndim=1] W):
        cdef int n,m,i,y,y1,t,a,b,a1,b1
        cdef double v,f,Z,h_e,h_v
        cdef int num_attrib_type = self.node_in_size
        n = self.sample.num_node
        m = self.sample.num_edge
        cdef np.ndarray[DTYPE_float_t, ndim=2] var_node_marginal
        cdef np.ndarray[DTYPE_float_t, ndim=3] factor_node_marginal
        var_node_marginal, factor_node_marginal = self.marginal_dict["log_likelihood"]

        f = 0.0  # f using no penalty form
        Z = 0.0
        #  \sum \lambda_i * f_i
        for i in range(n):
            h_v = 0.0
            y = self.sample.node_list[i].label  #监督信息
            for t in np.nonzero(x[i,:])[0]:
                f += W[self.get_attrib_parameter_id(y, t)] * x[i, t]
                for y1 in range(self.sample.num_label): #using Bethe Approximation
                    Z += W[self.get_attrib_parameter_id(y1, t)] * \
                        x[i, t] * var_node_marginal[i,y1]
            for a in range(self.sample.num_label):
                if fabs(var_node_marginal[i, a]) > 1e-10:
                    h_v += (-var_node_marginal[i, a] * log(var_node_marginal[i, a]))
            Z -= h_v * (len(self.factor_graph.var_node[i].neighbor)-1)

        for i in range(m):
            h_e = 0.0 # Edge entropy
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息, #FIXME 注意这里，用了数字形式的label
            f += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)]
            for a1 in range(self.sample.num_label):
                for b1 in range(self.sample.num_label):
                    Z += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1,b1)] * self.factor_graph.factor_node[i].marginal[a1,b1] # 多乘了一项marginal？这才是正确的，上面的似乎少乘了
                    if factor_node_marginal[i, a1,b1] > 1e-10:
                        h_e += (- factor_node_marginal[i, a1,b1] * log(factor_node_marginal[i, a1,b1]))
            Z += h_e
        f -= Z
        return f



    cpdef calc_gradient(self, np.ndarray[DTYPE_float_t, ndim=2] dx, np.ndarray[DTYPE_float_t, ndim=1] dw,
                        np.ndarray[DTYPE_float_t, ndim=2] x, np.ndarray[DTYPE_float_t, ndim=1] W, bint labeled_given):
        cdef int n, m,i,y,t,a,b,y_gt, a1,b1
        cdef int num_attrib_type = self.node_in_size
        n = self.sample.num_node
        m = self.sample.num_edge

        self.factor_graph.clear_data_for_sum_product()
        self.factor_graph.labeled_given = labeled_given
        cdef np.ndarray[DTYPE_float_t, ndim=2] var_node_marginal
        cdef np.ndarray[DTYPE_float_t, ndim=3] factor_node_marginal
        if labeled_given:
            var_node_marginal,factor_node_marginal = self.marginal_dict["labeled_given"]
        else:
            var_node_marginal,factor_node_marginal = self.marginal_dict["labeled_nogiven"]


        # dw shape = num_attrib x num_label + num_edge_type x num_label x num_label
        # dx shape = num_node x num_attrib
        for i in range(n):
            y_gt = self.sample.node_list[i].label  # ground truth supervise information
            for t in np.nonzero(x[i,:])[0]:
                dw[self.get_attrib_parameter_id(y_gt, t)] += x[i,t]
                dx[i,t] += W[self.get_attrib_parameter_id(y_gt, t)]
                for y in range(self.sample.num_label):
                    dw[self.get_attrib_parameter_id(y, t)] \
                        -= x[i, t] * var_node_marginal[i, y]   # note that y comes from ground truth, y1 comes from iterator over Y

                    dx[i,t] -= W[self.get_attrib_parameter_id(y, t)] * \
                                var_node_marginal[i,y]

        for i in range(m):
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息, #FIXME 注意这里，用了数字形式的label
            dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)] += 1

            for a1 in range(self.sample.num_label):
                for b1 in range(self.sample.num_label):
                    dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1, b1)] \
                        -= factor_node_marginal[i, a1, b1]







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


        print("forwarding before call parallel_bp")
        self.crf.parallel_bp(x,W)
        print("forwarding after call parallel_bp")
        f =0.0
        f = self.crf.log_likelihood(x, W)
        f *= -1.
        return utils.force_array(f, dtype=W.dtype),

cpdef crf_function( crf_pact_structure:CRFPackageStructure,x: chainer.Variable , W:chainer.Parameter, int node_in_size):
    x = F.copy(x, dst=-1)  # copy to host memory
    return CRFFunction(crf_pact_structure, node_in_size)(x, W)


