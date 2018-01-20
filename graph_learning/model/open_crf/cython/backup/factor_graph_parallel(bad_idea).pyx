#cython: profile=True

import hashlib
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, fabs
from multiprocessing import Pool
import functools

import os
# os.system("taskset -p 0xff %d" % os.getpid())

DTYPE_float = np.float32
DTYPE_int = np.int32

from copy import deepcopy

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
        # self.weight = p_lambda  # 这个p_lambda原来是偏移过的指针,FIXME，这句话要特别注意修改好
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
    cdef public str name
    def __init__(self):
        self.id = -1  # id is int. how to convert to str
        self.num_label = -1
        # neighbor and belief after add all_edge will become np.array 2d
        self.neighbor = []  # vector<Node*> neighbor
        self.neighbor_pos = dict() # neighbor => position (index in neighbor vector)
        self.msg = None # length == num_label
        # self.belief = [] # 这就是所谓信念，其实是二维数组，第一维是邻居个数，第二维是label_num（L）。邻居数量不定，为何第一个维度是vector原因也在于此
        self.belief_list = []
        self.belief = None
        self.name = "base node"

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
        if self.belief_list:  # 如果有的variable node没有邻居，就会造成belief=None
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

        # for y in range(self.num_label):
        #     if abs(self.belief[p][y] - msg_vec[y]) > diff_max.diff_max:
        #         diff_max.diff_max = abs(self.belief[p][y] - msg_vec[y])
        #     self.belief[p][y] = msg_vec[y]  #哈哈，为他的邻居设置belief为msg，不管该邻居是factor还是variable


cdef class VariableNode(Node):
    cdef public int y, label_type
    cdef public np.ndarray state_factor
    cdef public np.ndarray marginal
    def __init__(self):
        super(VariableNode, self).__init__()
        self.name = "var_node"  # must be called after super __init__
        self.y = -1 # label, may be np.array or true AU
        self.label_type = -1
        self.state_factor = None  # 长度是y，就是label个数的数组
        self.marginal = None  # variable node的marginal是长度为num_label的一维数组，而FactorNode的marginal是num_label x num_label的二维数组

    cpdef init(self, int num_label):
        super(VariableNode, self).basic_init(num_label)
        cdef np.ndarray[DTYPE_float_t, ndim=1] state_factor = np.zeros(num_label, dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=1] marginal = np.zeros(num_label, dtype=DTYPE_float)
        self.state_factor = state_factor
        self.marginal = marginal




    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    #FIXME, 写的有问题，这里的max_sum跟belief一样了
    cpdef max_sum_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef np.ndarray[DTYPE_float_t, ndim=1] product
        cdef FactorNode f
        cdef int i
        cdef np.ndarray all_index = np.arange(len(self.neighbor)).astype(np.int32)
        cdef np.ndarray[DTYPE_float_t, ndim=1] all_neighbor_product,filtered_index
        for i in range(len(self.neighbor)):
            f = self.neighbor[i]
            filtered_index = np.delete(all_index,i,0)
            all_neighbor_product = np.prod(np.asarray(self.belief)[filtered_index,:], axis=0,dtype=DTYPE_float)  # shape 1 x Y

            product = np.asarray(self.state_factor) * all_neighbor_product  # element-wise multiplication, 《PRML》 p406的公式(8.69)，因为一个factor node只有两个variable node邻居，去掉公式中连乘符号，公式里所谓的x其实就是label
            self.msg = product
            self.normalize_message()
            f.get_message_from(self.id, self.msg, diff_max)

    def __del__(self):
        if self.state_factor is not None:
            self.state_factor = None
        if self.marginal is not None:
            self.marginal = None


cdef class FactorNode(Node):
    cdef public FactorFunction func
    cdef public np.ndarray marginal
    cdef public np.ndarray func_values # key is np.nd
    def __init__(self):
        super(FactorNode, self).__init__()
        self.func = None
        self.marginal = None  # FactorNode的marginal是num_label x num_label的二维数组
        self.func_values = None  # cache for speed up belief_propagation
        self.name = "factor_node"  # must be called after super __init__



    cpdef init(self, int num_label):
        super(FactorNode, self).basic_init(num_label)
        cdef np.ndarray[DTYPE_float_t, ndim=2] marginal = np.zeros(shape=(num_label, num_label), dtype=DTYPE_float)
        self.marginal = marginal

    cpdef store_func_values(self,np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef np.ndarray[DTYPE_float_t, ndim=2] func_values = np.zeros((self.num_label,self.num_label),dtype=DTYPE_float)
        for y in range(self.num_label):
            for y1 in range(self.num_label):
                func_values[y, y1] = self.func.get_value(y, y1, weight)
        self.func_values = func_values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cpdef belief_propagation(self, np.ndarray[DTYPE_float_t, ndim=1] diff_neighbor_belief,
                             np.ndarray[DTYPE_float_t, ndim=1] diff_msg,
                             bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef int i,y,y1,p,msg_y
        cdef np.ndarray[DTYPE_float_t, ndim=2] func_values = self.func_values # cache for speed up
        cdef VariableNode var_neighbor
        cdef np.ndarray[DTYPE_float_t, ndim=2] msg = np.zeros((2, self.num_label), dtype=DTYPE_float)
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief = self.belief
        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief = np.zeros((2, self.num_label), dtype=DTYPE_float)

        for i in range(2):  # each factor node has only 2 neighbor variable node
            var_neighbor = self.neighbor[i]
            if labeled_given and var_neighbor.label_type == LabelTypeEnum.KNOWN_LABEL:
                msg_y = var_neighbor.y
                msg[i, msg_y] = 1.0
            else:
                msg[i, :] = np.dot(func_values, belief[1-i,:])  # <PRML> p404 Eqn.(8.66)
        msg /= np.sum(msg,axis=1).reshape(-1,1) # normalize, reshape for broadcast in numpy, this is mainly for boost speed
        for i in range(2):  # each factor node has only 2 neighbor variable node
            var_neighbor = self.neighbor[i]
            p = var_neighbor.neighbor_pos[self.id]
            neighbor_belief[i, :] = var_neighbor.belief[p,:]
            var_neighbor.belief[p, :] = msg[i,:]  # get_message_from msg

        diff_neighbor_belief[:2 * self.num_label] = neighbor_belief.ravel()
        diff_msg[:2 * self.num_label] = msg.ravel()



    cpdef max_sum_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1]  weight):
        cdef int i,y,y1
        cdef double s
        cdef int msg_index
        for i in range(2):
            if labeled_given and self.neighbor[i].label_type == LabelTypeEnum.KNOWN_LABEL:
                self.msg[:] = 0
                msg_index = self.neighbor[i].y
                self.msg[msg_index] = 1.0
            else:
                for y in range(self.num_label):
                    s = self.func.get_value(y, 0, weight) * self.belief[1-i, 0]
                    for y1 in range(self.num_label):
                        tmp = self.func.get_value(y, y1, weight) * self.belief[1-i, y1]
                        if tmp > s:
                            s = tmp # find max 《PRML》 p413 formular:（8.93）
                    self.msg[y] = s
                self.normalize_message()
            self.neighbor[i].get_message_from(self.id, self.msg, diff_max)

    def __del__(self):
        if self.marginal is not None:
            self.marginal = None

class Container(object):
    all_node = dict()
    @classmethod
    def clear(cls):
        Container.all_node.clear()

cpdef variable_bp_computation_unpack(args):
    return variable_bp_computation(*args)

#parallel only
cpdef variable_bp_computation(int variable_node_id,bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight,
                              int neighbor_num, np.ndarray[DTYPE_float_t, ndim=2] state_factor,
                              np.ndarray[DTYPE_float_t, ndim=2] _belief, int num_label):

        cdef FactorNode f
        cdef int i, p
        if neighbor_num == 0:
            return
        cdef np.ndarray[DTYPE_float_t, ndim=1] all_neighbor_product
        cdef np.ndarray[DTYPE_int_t, ndim=1] filter_idx  # optimize speed
        cdef np.ndarray[DTYPE_float_t, ndim=3] belief = np.tile(_belief, (neighbor_num, 1, 1)).astype(DTYPE_float)  # shape = neighbor_num x neighbor_num x Y
        cdef np.ndarray[DTYPE_float_t, ndim=2] _state_factor = state_factor.reshape(1, -1) # shape =1 x Y  # for broadcast multiplication
        belief[np.arange(belief.shape[0]), np.arange(belief.shape[1]), :] = 1.0
        cdef np.ndarray[DTYPE_float_t, ndim=2] msg = np.prod(belief, axis=1, dtype=DTYPE_float) * _state_factor  #shape = neighbor_num x Y. element-wise multiplication, <PRML> p406 Eqn. (8.69)
        msg /= np.sum(msg, axis=1).reshape(-1,1)  # normalize (reshape for broadcast)

        return neighbor_num, variable_node_id, msg, num_label
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef variable_bp_callback_unpack(args):
    assert len(args[0]) == 1,args
    variable_bp_callback(*args[0])
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef variable_bp_callback(int neighbor_num, int variable_node_id,
                           np.ndarray[DTYPE_float_t, ndim=2] msg, int num_label):
        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief = np.zeros((neighbor_num, num_label), dtype=DTYPE_float)
        cdef VariableNode variable_node = Container.all_node[variable_node_id]
        cdef FactorNode f
        cdef int p
        cdef int i
        for i in range(neighbor_num):
            f = variable_node.neighbor[i]
            p = f.neighbor_pos[variable_node.id]
            neighbor_belief[i, :] = f.belief[p,:]
            f.belief[p, :] = msg[i,:]  # get_message_from msg


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef factor_bp_computation_unpack(args):
    return factor_bp_computation(*args)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef factor_bp_computation(int factor_node_id, np.ndarray[DTYPE_float_t, ndim=2] func_values, int num_label,
                            np.ndarray[DTYPE_float_t, ndim=2] belief,
                            list neighbor_label_type, list neighbor_y, bint labeled_given):
        cdef int i,y,y1,p,msg_y
        cdef np.ndarray[DTYPE_float_t, ndim=2] msg = np.zeros((2, num_label), dtype=DTYPE_float)


        for i in range(2):  # each factor node has only 2 neighbor variable node
            if labeled_given and neighbor_label_type[i] == LabelTypeEnum.KNOWN_LABEL:
                msg_y = neighbor_y[i]
                msg[i, msg_y] = 1.0
            else:
                msg[i, :] = np.dot(func_values, belief[1-i,:])  # <PRML> p404 Eqn.(8.66)
        msg /= np.sum(msg,axis=1).reshape(-1,1) # normalize, reshape for broadcast in numpy, this is mainly for boost speed
        return factor_node_id, msg, num_label
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef factor_bp_callback_unpack(args):

    factor_bp_callback(*args[0])
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef factor_bp_callback(int factor_node_id,
                           np.ndarray[DTYPE_float_t, ndim=2] msg, int num_label):
        cdef np.ndarray[DTYPE_float_t, ndim=2] neighbor_belief = np.zeros((2, num_label), dtype=DTYPE_float)
        cdef FactorNode factor_node = Container.all_node[factor_node_id]
        cdef int i
        cdef VariableNode var_neighbor
        cdef int p
        for i in range(2):  # each factor node has only 2 neighbor variable node
            var_neighbor = factor_node.neighbor[i]
            p = var_neighbor.neighbor_pos[factor_node.id]
            neighbor_belief[i, :] = var_neighbor.belief[p,:]
            var_neighbor.belief[p, :] = msg[i,:]  # get_message_from msg
        #
        # diff_neighbor_belief[:2 * factor_node.num_label] = neighbor_belief.ravel()
        # diff_msg[:2 * factor_node.num_label] = msg.ravel()
#parallel part over

@cython.final
cdef class FactorGraph:


    # this is original InitGraph code
    def __init__(self, int n, int m, int num_label):
        cdef int p_node_id
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
        self.pool = Pool(processes=20)  # parallel matrix computation to speed up
        p_node_id = 0
        for i in range(self.n):
            self.var_node[i].id = p_node_id  # 这个id从0开始! 注意这个id不是Sample里的node的id
            self.p_node[p_node_id] = self.var_node[i]
            p_node_id += 1
            self.var_node[i].init(num_label)


        for i in range(self.m):
            self.factor_node[i].id = p_node_id # 注意这个id不是Sample里的node的id
            self.p_node[p_node_id] = self.factor_node[i]
            p_node_id += 1
            self.factor_node[i].init(num_label)
            # 注意还没有调用add_edge函数添加边


        self.factor_node_used = 0

        self.diff_max = DiffMax(diff_max=0.0)
        self.converged = False


    cpdef add_edge(self, int a, int b, FactorFunction func):
        '''
        :param a:  int type id
        :param b:  int type id
        :param func:
        :return:
        '''
        if self.factor_node_used == self.m:
            return
        # factor_node_used++ 是非常聪明的做法，因为其实每个factor_node(edge)没有区别，区别仅仅在function不一样
        self.factor_node[self.factor_node_used].func = func
        self.factor_node[self.factor_node_used].add_neighbor(self.var_node[a])
        self.factor_node[self.factor_node_used].add_neighbor(self.var_node[b])
        self.var_node[a].add_neighbor(self.factor_node[self.factor_node_used])
        self.var_node[b].add_neighbor(self.factor_node[self.factor_node_used])
        self.factor_node_used += 1

    cpdef add_edge_done(self):
        cdef int i
        cdef VariableNode var_node
        cdef Node p_node
        cdef np.ndarray[DTYPE_int_t, ndim=1] filtered_index, all_index

        for p_node in self.p_node:
            p_node.add_neighbor_done()
            self.all_diff_size += len(p_node.neighbor) * p_node.num_label



    cpdef clear_data_for_sum_product(self):
        for i in range(self.n):
            self.var_node[i].state_factor[:] = 1.0 / self.num_label
        for i in range(self.num_node):
            if self.p_node[i].neighbor:
                self.p_node[i].belief[:, :] = 1.0/self.num_label


    cpdef set_variable_label(self, int u, int y):
        self.var_node[u].y = y

    cpdef set_variable_state_factor(self, int u, int y, double v):
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
    cpdef belief_propagation(self, int max_iter, np.ndarray[DTYPE_float_t, ndim=1] weight):
        os.system("taskset -p 0xff {}".format(os.getpid()))
        cdef int start, end, dir,p,iter, i
        cdef FactorNode factor_node
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
        cdef list parallel_node_lst = []
        cdef list group
        for i in range(self.m):
            factor_node = self.factor_node[i]
            factor_node.store_func_values(weight)  # cache for boost speed
        for iter in range(max_iter):
            Container.clear()  # for parallel only
            parallel_node_lst.clear()
            current_start_diff = 0
            diff_max = 0.0
            if iter % 2 == 0: # 第偶数次迭代，从后向前传递消息，书上写了，再从根往回传，来回往复，num_node = n + m
                start = self.num_node-1
                end = -1
                dir = -1
            else:
                start = 0
                end = self.num_node
                dir = 1

            # last_name = None

            for p in range(start,end,dir):
                bfs_node = self.bfs_node[p]

                Container.all_node[bfs_node.id] = bfs_node
                # if last_name != bfs_node.name:
                #     parallel_node_lst.append([])
                #     last_name = bfs_node.name
                if bfs_node.name == "var_node":

                    parallel_node_lst.append([(bfs_node.id, self.labeled_given, weight, len(bfs_node.neighbor),
                                                  bfs_node.state_factor, bfs_node.belief, self.num_label)])
                elif bfs_node.name == "factor_node":
                    # print("append", bfs_node.name)
                    neighbor_label_type = [neighbor.label_type for neighbor in bfs_node.neighbor]
                    neighbor_y = [neighbor.y for neighbor in bfs_node.neighbor]
                    parallel_node_lst.append([(bfs_node.id,
                                                  bfs_node.func_values,self.num_label,bfs_node.belief, neighbor_label_type,
                                                  neighbor_y,self.labeled_given
                                                  )])
                current_start_diff += len(bfs_node.neighbor) * bfs_node.num_label
            for group in parallel_node_lst:
                if len(group) > 0:  # some node will neither be factor_node nor be variable node, strange?
                    type_name= Container.all_node[group[0][0]].name
                    if type_name == "var_node":
                        r = self.pool.map_async(variable_bp_computation_unpack,group,callback=variable_bp_callback_unpack)
                        r.wait()
                    elif type_name == "factor_node":
                        r = self.pool.map_async(factor_bp_computation_unpack, group,callback=factor_bp_callback_unpack)
                        r.wait()

            #     for bfs_node_tuple in group:
            #         bfs_node.belief_propagation(diff_neighbor_belief[current_start_diff:],diff_msg[current_start_diff:],
            #                                     self.labeled_given, weight) # 这个调用是override动态绑定，根据子类实现的不同调用不同的对象的方法
            #         current_start_diff += len(bfs_node.neighbor) * bfs_node.num_label

            # diff_max = np.max(diff_neighbor_belief - diff_msg)
            # if diff_max < 1e-6:
            #     break

    cpdef calculate_marginal(self, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef double sum_py, sump
        cdef int i,a,b
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief
        cdef np.ndarray[DTYPE_float_t, ndim=1] prod_result
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

    cpdef max_sum_propagation(self, int max_iter, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef int start,end, dir,p,iter
        start = 0
        end = 0
        dir = 0
        self.converged = False
        for iter in range(max_iter):
            self.diff_max.diff_max = 0
            if iter % 2 == 0:
                start = self.num_node - 1
                end = -1
                dir = -1
            else:
                start = 0
                end = self.num_node
                dir = 1
            for p in range(start, end, dir):
                self.bfs_node[p].max_sum_propagation(self.diff_max, self.labeled_given, weight)
            if self.diff_max.diff_max < 1e-6:
                break

    cpdef clean(self):
        if self.var_node:
            self.var_node.clear()
        if self.factor_node:
            self.factor_node.clear()
        self.bfs_node = None
        self.p_node = None

    def __del__(self):
        self.clean()
