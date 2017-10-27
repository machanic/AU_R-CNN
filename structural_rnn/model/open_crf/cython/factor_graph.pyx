
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, fabs
DTYPE_float = np.float32
DTYPE_int = np.int32
import time

cdef class DiffMax:

    def __init__(self,double diff_max):
        self.diff_max = diff_max
cpdef enum LabelTypeEnum:
    KNOWN_LABEL = 0
    UNKNOWN_LABEL = 1

cdef class FactorFunction:
    cpdef double get_value(self, int y1, int y2, np.ndarray[DTYPE_float_t, ndim=1] weight) except? -1:
        pass

cdef class EdgeFactorFunction(FactorFunction):
    def __init__(self, int num_label, int edge_type,
                 int num_edge_feature_each_type, int num_attrib_parameter,dict edge_feature_offset):
        super(EdgeFactorFunction, self).__init__()
        self.num_label = num_label
        self.feature_offset = edge_feature_offset
        self.edge_type = edge_type
        self.num_edge_feature_each_type = num_edge_feature_each_type
        self.num_attrib_parameter = num_attrib_parameter

    cpdef double get_value(self, int y1, int y2, np.ndarray[DTYPE_float_t, ndim=1] weight) except? -1:
        cdef int a,b,i
        cdef double value
        a = y1 if y1 < y2 else y2
        b = y1 if y1 > y2 else y2
        i = self.feature_offset[a * self.num_label + b]
        value = weight[self.num_attrib_parameter + self.edge_type * self.num_edge_feature_each_type + i]

        return exp(value)



cdef class Node:
    cdef public int id, num_label
    cdef public np.ndarray belief
    cdef public list belief_list, neighbor
    cdef public dict neighbor_pos
    cdef public np.ndarray msg
    def __init__(self):
        self.id = -1  # id is int. how to convert to str
        self.num_label = -1
        # neighbor and belief after add all_edge will become np.array 2d
        self.neighbor = []  # vector<Node*> neighbor
        self.neighbor_pos = dict() # neighbor => position (index in neighbor vector)
        self.msg = None # length == num_label
        self.belief_list = []
        self.belief = None # this is belief，actually 2-D array，0_th axis is neighbor number，2_nd axis is label_num（L）. for the reason neighbor number is varies in different node, thus why it is list

    cpdef init(self, int num_label):
        pass

    cpdef basic_init(self, int num_label):
        self.num_label = num_label
        cdef np.ndarray[DTYPE_float_t, ndim=1] msg = np.zeros(num_label, dtype=DTYPE_float)
        self.msg = msg

    def __del__(self):
        for arr in self.belief:
            arr = None
        self.belief = None
        self.msg = None

    cpdef add_neighbor(self, Node ng):
        self.neighbor_pos[ng.id] = len(self.neighbor)
        self.neighbor.append(ng)
        self.belief_list.append(np.zeros(self.num_label, dtype=DTYPE_float))

    cpdef add_neighbor_done(self):
        if self.belief_list:  # if some variable node does not have neighbor ,as a result belief=None
            self.belief = np.stack(self.belief_list).astype(dtype=DTYPE_float)

    cdef inline normalize_message(self):
        # cdef double s
        # cdef int y
        self.msg /= np.sum(self.msg)


    cpdef belief_propagation(self, np.ndarray[DTYPE_float_t, ndim=1] diff_neighbor_belief,
                             np.ndarray[DTYPE_float_t, ndim=1] diff_msg, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        pass


    cpdef max_sum_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        pass


    cdef get_message_from(self, int u, np.ndarray[DTYPE_float_t, ndim=1] msg_vec, DiffMax diff_max):
        cdef int p
        cdef double max_gap
        p = self.neighbor_pos[u]
        max_gap = fabs(np.max(self.belief[p,:] - msg_vec))
        if max_gap >  diff_max.diff_max:
            diff_max.diff_max = max_gap
        self.belief[p,:] = msg_vec
        # following is original code
        # for y in range(self.num_label):
        #     if abs(self.belief[p][y] - msg_vec[y]) > diff_max.diff_max:
        #         diff_max.diff_max = abs(self.belief[p][y] - msg_vec[y])
        #     self.belief[p][y] = msg_vec[y]  # yeah，set his neighbor belief=msg, no matter the neighbor is factor or variable


cdef class VariableNode(Node):
    cdef public int y, label_type
    cdef public np.ndarray state_factor
    cdef public np.ndarray marginal
    def __init__(self):
        super(VariableNode, self).__init__()
        self.y = -1 # label, may be np.array or true AU
        self.label_type = -1
        self.state_factor = None  # length=y， thus is label_num asarray
        self.marginal = None  # variable node is marginal, length = num_label 1-D array, while FactorNode's marginal is num_label x num_label 2-D array


    cpdef init(self, int num_label):
        super(VariableNode, self).basic_init(num_label)
        cdef np.ndarray[DTYPE_float_t, ndim=1] state_factor = np.zeros(num_label, dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=1] marginal = np.zeros(num_label, dtype=DTYPE_float)
        self.state_factor = state_factor
        self.marginal = marginal

    @cython.boundscheck(False)
    @cython.wraparound(False) # this optimized code , after this belief_propagation method, we show original python code
    cpdef belief_propagation(self, np.ndarray[DTYPE_float_t, ndim=1] diff_neighbor_belief,
                             np.ndarray[DTYPE_float_t, ndim=1] diff_msg,
                             bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef FactorNode f
        cdef int i, p
        cdef int neighbor_num = len(self.neighbor)
        if neighbor_num == 0:
            return
        cdef np.ndarray[DTYPE_float_t, ndim=1] all_neighbor_product
        cdef np.ndarray[DTYPE_int_t, ndim=1] filter_idx  # optimize speed
        cdef np.ndarray[DTYPE_float_t, ndim=3] belief = np.tile(self.belief, (neighbor_num, 1, 1)).astype(DTYPE_float)  # shape = neighbor_num x neighbor_num x Y
        cdef np.ndarray[DTYPE_float_t, ndim=2] state_factor = self.state_factor.reshape(1, -1) # shape =1 x Y  # for broadcast multiplication
        belief[np.arange(belief.shape[0]), np.arange(belief.shape[1]), :] = 1.0
        cdef np.ndarray[DTYPE_float_t, ndim=2] msg = np.prod(belief, axis=1, dtype=DTYPE_float) * state_factor  #shape = neighbor_num x Y. element-wise multiplication, <PRML> p406 Eqn. (8.69)
        msg /= np.sum(msg, axis=1).reshape(-1,1)  # normalize (reshape for broadcast)
        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief = np.zeros((neighbor_num, self.num_label), dtype=DTYPE_float)

        for i in range(neighbor_num):
            f = self.neighbor[i]
            p = f.neighbor_pos[self.id]
            neighbor_belief[i, :] = f.belief[p,:]
            f.belief[p, :] = msg[i,:]  # get_message_from msg

        diff_neighbor_belief[:neighbor_num*self.num_label] = neighbor_belief.ravel()
        diff_msg[: neighbor_num * self.num_label] = msg.ravel()

    '''
    # original python code, easy to understand but more slow
    def belief_propagation(self,diff_max:DiffMax, labeled_given:bool, weight:np.ndarray):
        global var_belief_call_num
        var_belief_call_num += 1
        product = 1.0
        for i in range(len(self.neighbor)):
            f = self.neighbor[i]  # factor node
            for y in range(self.num_label):
                product = self.state_factor[y]
                for j in range(len(self.neighbor)):
                    if i != j:
                        product *= self.belief[j][y]  # <PRML> p406 eqn. (8.69)，because one factor node only has 2 variable node neighbors，thus we omit \prod in original formular，the x in original formular is label
                self.msg[y] = product
            self.normalize_message()
            f.get_message_from(self.id, self.msg, diff_max)  # f is i_th neighbor, it is factor_node
    '''

    def __del__(self):
        if self.state_factor is not None:
            self.state_factor = None
        if self.marginal is not None:
            self.marginal = None


cdef class FactorNode(Node):
    cdef public int edge_type
    cdef public np.ndarray marginal
    cdef public np.ndarray func_value
    def __init__(self):
        super(FactorNode, self).__init__()
        self.edge_type = -1
        self.marginal = None  # FactorNode's marginal is num_label x num_label 2-D array
        self.func_value = None

    cpdef init(self, int num_label):
        super(FactorNode, self).basic_init(num_label)
        self.marginal = np.zeros(shape=(num_label, num_label), dtype=DTYPE_float)



    @cython.boundscheck(False)
    @cython.wraparound(False)  # this optimized code , after this belief_propagation method, we show original python code
    cpdef belief_propagation(self, np.ndarray[DTYPE_float_t, ndim=1] diff_neighbor_belief,
                             np.ndarray[DTYPE_float_t, ndim=1] diff_msg,
                             bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef int i,y,y1,p,msg_y
        cdef VariableNode var_neighbor
        cdef np.ndarray[DTYPE_float_t, ndim=2] msg = np.zeros((2, self.num_label), dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief = self.belief
        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief = np.zeros((2, self.num_label), dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=2] func_values = self.func_value

        for i in range(2):  # each factor node has only 2 neighbor variable node
            var_neighbor = self.neighbor[i]
            if labeled_given and var_neighbor.label_type == LabelTypeEnum.KNOWN_LABEL:
                msg_y = var_neighbor.y
                msg[i, msg_y] = 1.0
            else:
                msg[i, :] = np.dot(func_values, belief[1-i,:])  # <PRML> p404 Eqn.(8.66)
        msg /= np.sum(msg,axis=1).reshape(-1,1) # normalize, reshape for broadcast in numpy, this is mainly for boost speed
        for i in range(2):  # why we split to 2 times of loop, because np.sum only occur once will boost speed
            var_neighbor = self.neighbor[i]
            p = var_neighbor.neighbor_pos[self.id]
            neighbor_belief[i, :] = var_neighbor.belief[p,:]
            var_neighbor.belief[p, :] = msg[i,:]  # get_message_from msg

        diff_neighbor_belief[:2 * self.num_label] = neighbor_belief.ravel()
        diff_msg[:2 * self.num_label] = msg.ravel()
    '''
    # original python code, easy to understand but more slow
    def belief_propagation(self, diff_max:DiffMax, labeled_given:bool, weight:np.ndarray):
        for i in range(2):  # each factor node has only 2 neighbor variable node
            if labeled_given and self.neighbor[i].label_type == LabelTypeEnum.KNOWN_LABEL:
                self.msg[:] = 0
                self.msg[self.neighbor[i].y] = 1.0
            else:
                for y in range(self.num_label):
                    s = 0.0
                    for y1 in range(self.num_label):
                        s += self.func.get_value(y, y1, weight) * self.belief[1-i][y1]  # <PRML> p404 formular:8.66
                    self.msg[y] = s
                self.normalize_message()
            self.neighbor[i].get_message_from(self.id, self.msg, diff_max)  # neighbor[i]是variable_node
    '''


    def __del__(self):
        if self.marginal is not None:
            self.marginal = None


@cython.final
cdef class FactorGraph:


    # this is original InitGraph code
    def __init__(self, int n, int m, int num_label, list func_list=None):
        
        self.labeled_given = False
        self.n = n
        self.m = m
        self.num_label = num_label
        self.num_node = n + m
        self.var_node = [VariableNode() for _ in range(self.n)]  #  本来代码是n个长度的var_node数组， key is node_id, which is frame_boxid
        self.factor_node = [FactorNode() for _ in range(self.m)]
        self.bfs_node = np.empty(self.num_node, dtype=object)
        self.p_node = np.empty(self.num_node, dtype=object) # p_node contains all node
        self.all_diff_size = 0
        self.edge_type_func_list = func_list
        self.diff_max = DiffMax(diff_max=0.0)
        self.converged = False


    cpdef add_edge(self, int factor_node_index, int a, int b, int edge_type):
        '''
        :param factor_node_index: the same with index in self.factor_node (0...m-1), but not factor_node.id
        :param a:  int type id
        :param b:  int type id
        :param func:
        :return:
        '''

        # factor_node_used++ is very clever way，for the reason that each factor_node(edge) has no different，only differs in its func
        self.factor_node[factor_node_index].edge_type = edge_type
        self.factor_node[factor_node_index].add_neighbor(self.var_node[a])
        self.factor_node[factor_node_index].add_neighbor(self.var_node[b])
        self.var_node[a].add_neighbor(self.factor_node[factor_node_index])
        self.var_node[b].add_neighbor(self.factor_node[factor_node_index])

    cpdef add_edge_done(self):
        cdef int i, edge_type
        cdef Node p_node
        cdef np.ndarray[DTYPE_int_t, ndim=1] filtered_index, all_index

        for p_node in self.p_node:
            p_node.add_neighbor_done() # convert to np.ndarray
            self.all_diff_size += len(p_node.neighbor) * p_node.num_label



    cpdef clear_data_for_sum_product(self):
        for i in range(self.n):
            self.var_node[i].state_factor[:] = 1.0 / self.num_label
        for i in range(self.num_node):
            if self.p_node[i].neighbor:
                self.p_node[i].belief[:, :] = 1.0/self.num_label


    cpdef set_variable_label(self, int u, int y):
        self.var_node[u].y = y

    cpdef set_variable_state_factor(self, int u, np.ndarray[DTYPE_float_t, ndim=1] state_factor):
        self.var_node[u].state_factor = state_factor

    # deprecated
    cpdef set_variable_state_factor_y(self, int u, int y, double v):
        self.var_node[u].state_factor[y] = v

    cpdef gen_propagate_order(self):
        mark = np.full(shape=self.num_node, fill_value=False, dtype=bool) # visit mark
        cdef np.ndarray[object, ndim=1] bfs_node = np.empty(self.num_node, dtype=object)
        self.bfs_node = bfs_node
        cdef int head, tail, i
        head = 0
        tail = -1
        for i in range(self.num_node):

            if mark[i] == False:
                tail += 1
                self.bfs_node[tail] = self.p_node[i]
                mark[self.p_node[i].id] = True
                while head <= tail:
                    u = self.bfs_node[head]
                    head += 1
                    for it in u.neighbor:
                        if mark[it.id] == False:
                            tail += 1
                            self.bfs_node[tail] = it
                            mark[it.id] = True
        mark = None
        del mark

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef belief_propagation(self, int max_iter, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef int start, end, dir,p,iter, i,y,y1,edge_type
        cdef FactorNode factor_node
        cdef FactorFunction func
        start = 0
        end = 0
        dir = 0
        self.converged = False
        cdef Node bfs_node
        cdef double diff_max = 0.0
        cdef double current_max
        cdef np.ndarray[DTYPE_float_t, ndim=1] diff_neighbor_belief = np.zeros(self.all_diff_size, dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=1] diff_msg = np.zeros(self.all_diff_size, dtype=DTYPE_float)
        cdef int current_start_diff
        cdef dict func_value_dict = dict()
        cdef np.ndarray[DTYPE_float_t, ndim=2] func_value

        # cache for boost speed
        for edge_type, func in enumerate(self.edge_type_func_list):
            func_value = np.zeros((self.num_label,self.num_label),dtype=DTYPE_float)
            for y in range(self.num_label):
                for y1 in range(self.num_label):
                    func_value[y, y1] = func.get_value(y, y1, weight)
            func_value_dict[edge_type] = func_value
        for factor_node in self.factor_node:
            factor_node.func_value = func_value_dict[factor_node.edge_type]

        for iter in range(max_iter):
            current_start_diff = 0
            diff_max = 0.0
            if iter % 2 == 0:  # even times from last pass message to first，<PRML> says: then from root to leaf and next time from leaf to root over again，num_node = n + m
                start = self.num_node-1
                end = -1
                dir = -1
            else:
                start = 0
                end = self.num_node
                dir = 1
            for p in range(start, end, dir):
                bfs_node = self.bfs_node[p]
                bfs_node.belief_propagation(diff_neighbor_belief[current_start_diff:],diff_msg[current_start_diff:],
                                            self.labeled_given, weight) # override which function will be called determined runtime(which Node subclass are using)
                current_start_diff += len(bfs_node.neighbor) * bfs_node.num_label

            diff_max = np.max(diff_neighbor_belief - diff_msg)
            if diff_max < 1e-6:
                break

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef calculate_marginal(self, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef double sum_py, sump
        cdef int i,a,b,y,y1,edge_type
        cdef FactorFunction func
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief
        cdef np.ndarray[DTYPE_float_t, ndim=1] prod_result
        cdef np.ndarray[DTYPE_float_t, ndim=2] func_value
        cdef VariableNode var_node
        for i in range(self.n):
            var_node = self.var_node[i]
            belief = var_node.belief
            prod_result = np.ones(shape=self.num_label, dtype=DTYPE_float)
            if belief is not None:
                prod_result = np.prod(belief, axis=0, dtype=DTYPE_float)  # shape = (Y,)
            var_node.marginal = var_node.state_factor * prod_result
            sum_py = np.sum(var_node.marginal)
            var_node.marginal /= sum_py

        # prepare cache the func_value to speed up because Y is large, we cannot let it occur inside for-loop of self.m
        cdef dict func_value_dict = dict()
        for edge_type, func in enumerate(self.edge_type_func_list):
            func_value = np.zeros((self.num_label,self.num_label),dtype=DTYPE_float)
            for y in range(self.num_label):
                for y1 in range(self.num_label):
                    func_value[y, y1] = func.get_value(y, y1, weight)
            func_value_dict[edge_type] = func_value

        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief_0 = np.zeros(shape=(self.m, self.num_label), dtype=DTYPE_float) # shape = (m, Y), where Y denotes num_label
        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief_1 = np.zeros(shape=(self.m, self.num_label), dtype=DTYPE_float) # shape = (m, Y)
        for i in range(self.m):  # in order to use np.einsum without for-loop, we divide for m-loop into 2 parts
            neighbor_belief_0[i, :] = self.factor_node[i].belief[0,:]
            neighbor_belief_1[i, :] = self.factor_node[i].belief[1,:]
        cdef np.ndarray[DTYPE_float_t, ndim=3] neighbor_belief = np.einsum('ki,kj->kij', neighbor_belief_0, neighbor_belief_1)  # shape = (m, Y, Y) consume memory if Y is large
        cdef np.ndarray[DTYPE_float_t, ndim=2] factor_node_marginal
        for i in range(self.m):
            factor_node_marginal = self.factor_node[i].marginal
            factor_node_marginal += neighbor_belief[i,:,:] * func_value_dict[self.factor_node[i].edge_type]
            factor_node_marginal /= np.sum(factor_node_marginal)

        ''' original code is below, we split to two times of loops, because we want to use np.einsum only once, which we consume more memory but will speed up by numpy
        cdef np.ndarray[DTYPE_float_t, ndim=2] factor_node_marginal
        for i in range(self.m):
            factor_node_marginal = self.factor_node[i].marginal
            sump = 0.0
            for a in range(self.num_label):
                for b in range(self.num_label):
                    factor_node_marginal[a, b] += \
                        self.factor_node[i].belief[0, a] \
                        * self.factor_node[i].belief[1, b] \
                        * self.factor_node[i].func.get_value(a, b, weight)
                    sump += factor_node_marginal[a, b]
            factor_node_marginal /= sump
        '''


    cpdef clean(self):
        if self.var_node:
            self.var_node.clear()
        if self.factor_node:
            self.factor_node.clear()
        self.bfs_node = None
        self.p_node = None

    def __del__(self):
        self.clean()
