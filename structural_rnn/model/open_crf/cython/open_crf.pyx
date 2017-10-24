
import numpy as np
cimport numpy as np
import cython
DTYPE_float = np.float32
DTYPE_int = np.int32
DTYPE_obj = np.object
ctypedef np.float32_t DTYPE_float_t
ctypedef np.int32_t DTYPE_int_t
from libc.math cimport exp, fabs,log,sqrt
from structural_rnn.openblas_toolkit.openblas_configure import num_threads

import chainer
from chainer import Function
from chainer import utils
from chainer.utils import type_check
from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
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
    cdef public int node_in_size, num_label
    cdef public dict edge_parameter_id_dict
    cdef public list edge_parameter_index_lst
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
        self.edge_parameter_id_dict = <dict>self.get_edge_parameter_id_dict()  # cache all get_edge_parameter_id to boost speed

        

    cdef object get_edge_parameter_id_dict(self):
        '''
            this method mainly to cache all get_edge_parameter_id to boost speed 
        '''
        cdef int edge_type,a,b
        edge_parameter_id_dict = defaultdict(list)
        for edge_type in range(self.num_edge_type):
            for a in range(self.num_label):
                for b in range(self.num_label):
                    edge_parameter_id_dict[edge_type].append(self.get_edge_parameter_id(edge_type,a,b))
        return edge_parameter_id_dict



    cdef int get_attrib_parameter_id(self, int y, int x):
        return y * self.node_in_size + x

    cdef int get_edge_parameter_id(self, int edge_type, int a, int b):
        cdef int key, offset
        key = self.num_label * (a if a < b else b) + (a if a > b else b)
        offset = self.edge_feature_offset[key]
        return self.num_attrib_parameter + edge_type * self.num_edge_feature_each_type + offset

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double log_likelihood(self, np.ndarray[DTYPE_float_t, ndim=2] x,
                                np.ndarray[DTYPE_float_t, ndim=1] W):
        # print("enter in log_likelihood")
        cdef int n,m,i,y,y1,a,b,a1,b1,t, edge_type, gt_edge_type
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
        '''
        below is original code, it is easy to read but too slow , we use np.dot to optimize it
        for i in range(n):
            for y in range(self.num_label):
                v = 0.0
                for t in self.x_nonzero_attrib[i]:
                    v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
                self.factor_graph.set_variable_state_factor_y(i, y, exp(v))  #得到紧致表达
        # we must compute marginal after belief_propagation
        '''
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        self.factor_graph.calculate_marginal(W)  # this line in my machine, BP4D database takes approximately 56 secs
        

        f = 0.0  # f using no penalty form
        Z = 0.0
        #  \sum \lambda_i * f_i
        for i in range(n):
            h_v = 0.0
            y = self.sample.node_list[i].label  #supervise information
            f += WX[i,y]
            # Z += np.einsum('yt,t,y->',W_1, x[i,:], self.factor_graph.var_node[i].marginal) because WX has already computed , thus following line will substitute this line
            Z +=  np.dot(WX[i,:] , self.factor_graph.var_node[i].marginal) #using Bethe Approximation
            # we use np.einsum to speed up following code
            # for t in range(len(x[i])):
            #     for y1 in range(self.num_label): #using Bethe Approximation
            #         Z += W[self.get_attrib_parameter_id(y1, t)] * x[i, t] * self.factor_graph.var_node[i].marginal[y1]
            h_v += (-np.dot(self.factor_graph.var_node[i].marginal, np.log(self.factor_graph.var_node[i].marginal)))
            Z -= h_v * (len(self.factor_graph.var_node[i].neighbor)-1)

        cdef np.ndarray[DTYPE_float_t, ndim=3] all_factor_marginal = np.zeros(shape=(m, self.num_label, self.num_label), dtype=DTYPE_float)
        h_e = 0.0 # Edge entropy
        for i in range(m):
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #supervise information
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #supervise information note that we use combine label as one int value
            gt_edge_type = self.sample.edge_list[i].edge_type
            f += W[self.get_edge_parameter_id(gt_edge_type, a, b)]
            Z += np.dot(W[self.edge_parameter_id_dict[gt_edge_type]], self.factor_graph.factor_node[i].marginal.ravel())
            all_factor_marginal[i, :, :] = self.factor_graph.factor_node[i].marginal
            '''
            we want to optimize following code, when num_label is large, for instance = 600, using any operator inside m-loop will retard the program
            we decide to exchange space for time, use large 3-D array to store marginal, then calculate it once
            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    Z += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1,b1)] * self.factor_graph.factor_node[i].marginal[a1,b1] # 多乘了一项marginal？这才是正确的，上面的似乎少乘了
                    if self.factor_graph.factor_node[i].marginal[a1,b1] > 1e-10:
                        h_e += (- self.factor_graph.factor_node[i].marginal[a1,b1] * log(self.factor_graph.factor_node[i].marginal[a1,b1]))
            Z += h_e
            '''

        h_e += (-np.einsum('mij,mij->',all_factor_marginal, np.log(all_factor_marginal)))
        Z += h_e
        f -= Z

        '''
        below is original code, which is easy to understand but slow to run.
        for i in range(n):
            h_v = 0.0
            y = self.sample.node_list[i].label  #supervise information
            for t in range(len(x[i])):
                f += W[self.get_attrib_parameter_id(y, t)] * x[i, t]
                for y1 in range(self.num_label): #using Bethe Approximation
                    Z += W[self.get_attrib_parameter_id(y1, t)] * x[i, t] * self.factor_graph.var_node[i].marginal[y1]
            for a in range(self.num_label):
                if fabs(self.factor_graph.var_node[i].marginal[a]) > 1e-10:
                        h_v += (-self.factor_graph.var_node[i].marginal[a] * log(self.factor_graph.var_node[i].marginal[a]))
            Z -= h_v * (len(self.factor_graph.var_node[i].neighbor)-1)

        for i in range(m):
            h_e = 0.0 # Edge entropy
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #supervise information
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #supervise information note that we use combine label as one int value
            f += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)]
            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    Z += W[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1,b1)] * self.factor_graph.factor_node[i].marginal[a1,b1] # 多乘了一项marginal？这才是正确的，上面的似乎少乘了
                    if self.factor_graph.factor_node[i].marginal[a1,b1] > 1e-10:
                        h_e += (- self.factor_graph.factor_node[i].marginal[a1,b1] * log(self.factor_graph.factor_node[i].marginal[a1,b1]))
            Z += h_e
        f -= Z
        '''

        return f



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef calc_gradient(self, np.ndarray[DTYPE_float_t, ndim=2] dx, np.ndarray[DTYPE_float_t, ndim=1] dw,
                        np.ndarray[DTYPE_float_t, ndim=2] x, np.ndarray[DTYPE_float_t, ndim=1] W, bint labeled_given):
        cdef int n, m,i,y,t,a,b,y_gt
        cdef int num_attrib_type = self.node_in_size
        n = self.sample.num_node
        m = self.sample.num_edge

        self.factor_graph.clear_data_for_sum_product()
        self.factor_graph.labeled_given = labeled_given

        cdef np.ndarray[DTYPE_float_t, ndim=2] W_1 = W[0:self.num_label*num_attrib_type].reshape(self.num_label, num_attrib_type)
        cdef np.ndarray[DTYPE_float_t, ndim=2] WX = np.dot(W_1, x.T).T # W_1 shape= Y x t ; x.T shape = t x n ==> result matrix = Y x n ==> transpose n x Y
        cdef np.ndarray[DTYPE_float_t, ndim=2] variable_state_factor = np.exp(WX)
        for i in range(n):
            self.factor_graph.set_variable_state_factor(i, variable_state_factor[i,:])
        '''
        for i in range(n):
            for y in range(self.num_label):
                v = 0.0
                for t in range(len(x[i])):
                    v += W[y * num_attrib_type + t] * x[i, t] # 将self.sample.node_list[i].feature[t]改为x[i,t]
                self.factor_graph.set_variable_state_factor_y(i, y, exp(v))  #得到紧致表达
        '''
        # print("in calc_gradient, belief_propagation begin")
        # start = time.time()
        self.factor_graph.belief_propagation(self.max_bp_iter, W)
        # end = time.time()
        # print("in calc_gradient, belief_propagation takes {}".format(end-start))
        # start = time.time()
        self.factor_graph.calculate_marginal(W)
        # end = time.time()
        # print("in calc_gradient calculate_marginal takes {}".format(end-start))
        # start = time.time()
        # dw shape = num_attrib x num_label + num_edge_type x num_label x num_label
        # dx shape = num_node x num_attrib
        cdef np.ndarray[DTYPE_float_t, ndim=2] dw_1 = dw[0: self.num_label * num_attrib_type].reshape(self.num_label, num_attrib_type)  # change value of dw_1 will also change value of dw
        cdef np.ndarray[DTYPE_float_t, ndim=2] var_prob = np.zeros(shape=(n, self.num_label), dtype=DTYPE_float)
        for i in range(n):
            y_gt = self.sample.node_list[i].label  # ground truth supervise information
            dw_1[y_gt, :] += x[i,:] # trick of boost speed
            dx[i, :] += W_1[y_gt, :]  # trick of boost speed
            var_prob[i, :] = self.factor_graph.var_node[i].marginal  # trick of boost speed
        dw_1 -= np.dot(var_prob.T, x)  # (Y x n) x (n x t)
        dx -= np.dot(var_prob, W_1) # (n x Y) x (Y x t) trick of boost speed
        # end = time.time()
        # print("in calc_gradient, n-loop(1) takes {}".format(end- start))

        '''
        below is original code, we use numpy combine operate to boost speed
        for i in range(n):
            y_gt = self.sample.node_list[i].label  # ground truth supervise information
            for t in range(len(x[i])):
                dw[self.get_attrib_parameter_id(y_gt, t)] += x[i,t]
                dx[i,t] += W[self.get_attrib_parameter_id(y_gt, t)]
                for y in range(self.num_label):
                    dw[self.get_attrib_parameter_id(y, t)] \
                        -= x[i, t] * self.factor_graph.var_node[i].marginal[y]   # note that y comes from ground truth, y1 comes from iterator over Y

                    dx[i,t] -= W[self.get_attrib_parameter_id(y, t)] * \
                                self.factor_graph.var_node[i].marginal[y]
        '''


        cdef int gt_edge_type
        # start = time.time()
        for i in range(m):
            # if i % 1000 == 0:  # each iteration will take more time is label_number became large
            #     end = time.time()
            #     print("m({0}/{1}={2}) in 1000:{3}".format(i,m,float(i)/m, end-start))
            #     start = time.time()
            a = self.sample.node_list[self.sample.edge_list[i].a].label  #supervise information
            b = self.sample.node_list[self.sample.edge_list[i].b].label  #supervise information, notice that we use combine label as one int
            dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a, b)] += 1
            gt_edge_type = self.sample.edge_list[i].edge_type
            # below code works fine if num_label is not large, but when num_label is large, still slow, thus we use new trick: np.bincount
            # np.subtract.at(dw, self.edge_parameter_id_dict[gt_edge_type], self.factor_graph.factor_node[i].marginal.ravel())
            dw -= np.bincount(self.edge_parameter_id_dict[gt_edge_type], self.factor_graph.factor_node[i].marginal.ravel(), minlength=dw.size).astype(dw.dtype)

            '''
            below is original code: because when num_label is large, this double-for-loop will consume much time, notice that 
            a1, b1 the same index value may occur multiple times, for instance a1=1,b1=2 and a1=2,b1=1 will get_edge_parameter_id produce same index
            thus we need np.subtract.at
            for a1 in range(self.num_label):
                for b1 in range(self.num_label):
                    dw[self.get_edge_parameter_id(self.sample.edge_list[i].edge_type, a1, b1)] \
                        -= self.factor_graph.factor_node[i].marginal[a1, b1]
            '''
        # end = time.time()
        # print("after m(2) loop:{}".format(end-start))






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
        f = self.crf.log_likelihood(x, W)
        f *= -1.
        return utils.force_array(f, dtype=W.dtype),

cpdef crf_function( crf_pact_structure:CRFPackageStructure,x: chainer.Variable , W:chainer.Parameter, int node_in_size):
    x = F.copy(x, dst=-1)  # copy to host memory
    return CRFFunction(crf_pact_structure, node_in_size)(x, W)


