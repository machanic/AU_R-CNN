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



cdef class CRF_CFunction:
    cdef public FactorGraph factor_graph
    cdef public int max_bp_iter, num_feature, num_attrib_parameter, num_edge_feature_each_type, num_attrib_type
    cdef dict edge_feature_offset
    cdef public dict x_nonzero_attrib
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
        self.x_nonzero_attrib = <dict>defaultdict(list)



    cdef int get_attrib_parameter_id(self, int y, int x):
        return y * self.node_in_size + x

    cdef int get_edge_parameter_id(self, int edge_type, int a, int b):
        cdef int key
        key = self.num_label * (a if a < b else b) + (a if a > b else b)
        offset = self.edge_feature_offset[key]
        return self.num_attrib_parameter + edge_type * self.num_edge_feature_each_type + offset

    cpdef double log_likelihood(self, np.ndarray[DTYPE_float_t, ndim=2] x,
                                np.ndarray[DTYPE_float_t, ndim=1] W):
        print("in log_likelihood")
        cdef int n,m,i,y,y1,t,a,b,a1,b1
        cdef double v,f,Z,h_e,h_v
        cdef int num_attrib_type = self.node_in_size
        n = self.sample.num_node
        m = self.sample.num_edge

        self.factor_graph.labeled_given = False
        self.factor_graph.clear_data_for_sum_product()
        print("before factor_graph set_variable_state_factor") # FIXME
        for i in range(n):
            for y in range(self.factor_graph.num_label):
                v = 0.0
                for t in self.x_nonzero_attrib[i]:
                    v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
                self.factor_graph.set_variable_state_factor(i, y, exp(v))  #得到紧致表达
        # 需要算出marginal
        print("before factor_graph belief propagation") # FIXME
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        print("before factor_graph calculate_marginal") # FIXME
        self.factor_graph.calculate_marginal(W)
        print("after factor_graph calculate_marginal") # FIXME
        f = 0.0  # f using no penalty form
        Z = 0.0
        #  \sum \lambda_i * f_i
        for i in range(n):
            h_v = 0.0
            y = self.sample.node_list[i].label  #监督信息
            for t in self.x_nonzero_attrib[i]:
                f += W[self.get_attrib_parameter_id(y, t)] * x[i, t]
                for y1 in range(self.num_label): #using Bethe Approximation
                    Z += W[self.get_attrib_parameter_id(y1, t)] * \
                        x[i, t] * self.factor_graph.var_node[i].marginal[y1]
            for a in range(self.num_label):
                if fabs(self.factor_graph.var_node[i].marginal[a]) > 1e-10:
                    h_v += (-self.factor_graph.var_node[i].marginal[a] * log(self.factor_graph.var_node[i].marginal[a]))
            Z -= h_v * (len(self.factor_graph.var_node[i].neighbor)-1)

        for i in range(m):
            h_e = 0.0 # Edge entropy
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息, #FIXME 注意这里，用了数字形式的label
            f += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)]
            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    Z += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1,b1)] * self.factor_graph.factor_node[i].marginal[a1,b1] # 多乘了一项marginal？这才是正确的，上面的似乎少乘了
                    if self.factor_graph.factor_node[i].marginal[a1,b1] > 1e-10:
                        h_e += (- self.factor_graph.factor_node[i].marginal[a1,b1] * log(self.factor_graph.factor_node[i].marginal[a1,b1]))
            Z += h_e
        f -= Z
        return f



    cpdef calc_gradient(self, np.ndarray[DTYPE_float_t, ndim=2] dx, np.ndarray[DTYPE_float_t, ndim=1] dw,
                        np.ndarray[DTYPE_float_t, ndim=2] x, np.ndarray[DTYPE_float_t, ndim=1] W, bint labeled_given):
        print("in calc_gradient")
        cdef int n, m,i,y,t,a,b,y_gt, a1,b1
        cdef int num_attrib_type = self.node_in_size
        n = self.sample.num_node
        m = self.sample.num_edge
        self.factor_graph.labeled_given = labeled_given
        self.factor_graph.clear_data_for_sum_product()

        for i in range(n):
            for y in range(self.factor_graph.num_label):
                v = 0.0
                for t in self.x_nonzero_attrib[i]:
                    v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
                self.factor_graph.set_variable_state_factor(i, y, exp(v))  #得到紧致表达
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        self.factor_graph.calculate_marginal(W)
        # dw shape = num_attrib x num_label + num_edge_type x num_label x num_label
        # dx shape = num_node x num_attrib
        for i in range(n):
            y_gt = self.sample.node_list[i].label  # ground truth supervise information
            for t in self.x_nonzero_attrib[i]:
                dw[self.get_attrib_parameter_id(y_gt, t)] += x[i,t]
                dx[i,t] += W[self.get_attrib_parameter_id(y_gt, t)]
                for y in range(self.num_label):
                    dw[self.get_attrib_parameter_id(y, t)] \
                        -= x[i, t] * self.factor_graph.var_node[i].marginal[y]   # note that y comes from ground truth, y1 comes from iterator over Y

                    dx[i,t] -= W[self.get_attrib_parameter_id(y, t)] * \
                                self.factor_graph.var_node[i].marginal[y]

        for i in range(m):
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #监督信息
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #监督信息, #FIXME 注意这里，用了数字形式的label
            dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)] += 1

            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1, b1)] \
                        -= self.factor_graph.factor_node[i].marginal[a1, b1]







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
        nonzero_tuple = np.nonzero(x)
        cdef int i,j
        for i,j in zip(*nonzero_tuple):
            self.crf.x_nonzero_attrib[i].append(j)
        f =0.0
        with num_threads(8):
            f = self.crf.log_likelihood(x, W)
            f *= -1.
        return utils.force_array(f, dtype=W.dtype),

cpdef crf_function( crf_pact_structure:CRFPackageStructure,x: chainer.Variable , W:chainer.Parameter, int node_in_size):
    x = F.copy(x, dst=-1)  # copy to host memory
    return CRFFunction(crf_pact_structure, node_in_size)(x, W)


