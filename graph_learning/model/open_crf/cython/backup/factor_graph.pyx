


import numpy as np
cimport numpy as np
cimport cython

DTYPE_float = np.float32
DTYPE_int = np.int32

cdef extern from "math.h":
    double exp(double x)
    double fabs(double x)



cdef class DiffMax:

    def __init__(self,double diff_max):
        self.diff_max = diff_max


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
    cdef public list belief_list,neighbor
    cdef public dict neighbor_pos
    cdef public np.ndarray msg
    def __init__(self):
        cdef dict neighbor_pos = dict()
        self.id = -1  # id is int. how to convert to str
        self.num_label = -1
        # neighbor and belief after add all_edge will become np.array 2d
        self.neighbor = []  # vector<Node*> neighbor
        self.neighbor_pos = neighbor_pos # neighbor => position (index in neighbor vector)
        self.msg = None # length == num_label
        # self.belief = [] # 这就是所谓信念，其实是二维数组，第一维是邻居个数，第二维是label_num（L）。邻居数量不定，为何第一个维度是vector原因也在于此
        self.belief_list = []
        self.belief = None

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
        cdef np.ndarray[DTYPE_float_t, ndim=1] each_belief = np.zeros(self.num_label, dtype=DTYPE_float)
        self.neighbor_pos[ng.id] = len(self.neighbor)
        self.neighbor.append(ng)
        self.belief_list.append(each_belief)

    cpdef add_neighbor_done(self):
        if self.belief_list:  # 如果有的variable node没有邻居，就会造成belief=None
            self.belief = np.stack(self.belief_list)



    cdef inline normalize_message(self):
        # cdef double s
        # cdef int y
        self.msg /= np.sum(self.msg)


    cpdef belief_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        pass


    cpdef max_sum_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        pass


    cdef get_message_from(self, int u, np.ndarray[DTYPE_float_t, ndim=1] msg_vec, DiffMax diff_max):
        cdef int p
        cdef double max_gap
        p = self.neighbor_pos[u]
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief = self.belief
        max_gap = fabs(np.max(belief[p,:] - msg_vec))
        if max_gap >  diff_max.diff_max:
            diff_max.diff_max = max_gap
        belief[p,:] = msg_vec

        # for y in range(self.num_label):
        #     if abs(self.belief[p][y] - msg_vec[y]) > diff_max.diff_max:
        #         diff_max.diff_max = abs(self.belief[p][y] - msg_vec[y])
        #     self.belief[p][y] = msg_vec[y]  #哈哈，为他的邻居设置belief为msg，不管该邻居是factor还是variable


cdef class VariableNode(Node):
    cdef public int y, label_type
    cdef public np.ndarray state_factor
    cdef public np.ndarray marginal
    cdef public list filtered_index_list
    def __init__(self):
        super(VariableNode, self).__init__()
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


    cpdef belief_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef np.ndarray[DTYPE_float_t, ndim=1] product
        cdef FactorNode f
        cdef int i
        # cdef np.ndarray[DTYPE_int_t, ndim=1] all_index = np.arange(len(self.neighbor)).astype(np.int32)
        cdef np.ndarray[DTYPE_float_t, ndim=1] all_neighbor_product
        cdef np.ndarray[DTYPE_int_t, ndim=1] filter_idx  # optimize speed
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief = self.belief
        cdef np.ndarray[DTYPE_float_t, ndim=1] state_factor = self.state_factor
        for i in range(len(self.neighbor)):
            f = self.neighbor[i]
            # filtered_index = np.delete(all_index,i,0)  # old code
            filter_idx = self.filtered_index_list[i]  # optimize speed
            all_neighbor_product = np.prod(belief[filter_idx,:], axis=0,dtype=DTYPE_float)  # shape 1 x Y

            product = state_factor * all_neighbor_product  # element-wise multiplication, <PRML> p406 Eqn. (8.69)，因为一个factor node只有两个variable node邻居，去掉公式中连乘符号，公式里所谓的x其实就是label
            self.msg = product
            self.normalize_message()
            f.get_message_from(self.id, self.msg, diff_max)


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
    def __init__(self):
        super(FactorNode, self).__init__()
        self.func = None
        self.marginal = None  # FactorNode的marginal是num_label x num_label的二维数组


    cpdef init(self, int num_label):
        super(FactorNode, self).basic_init(num_label)
        cdef np.ndarray[DTYPE_float_t, ndim=2] marginal = np.zeros(shape=(num_label, num_label), dtype=DTYPE_float)
        self.marginal = marginal


    cpdef belief_propagation(self, DiffMax diff_max, bint labeled_given, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef int i,y,y1
        cdef np.ndarray[DTYPE_float_t, ndim=2] func_values = np.zeros(shape=(self.num_label,self.num_label),dtype=DTYPE_float)
        cdef int msg_y
        cdef VariableNode var_neighbor
        cdef np.ndarray[DTYPE_float_t, ndim=2] msg = np.zeros((2, self.num_label), dtype=np.float32)
        cdef np.ndarray[DTYPE_float_t, ndim=2] belief = self.belief
        # cache for speed up
        for y in range(self.num_label):
            for y1 in range(self.num_label):
                func_values[y, y1] = self.func.get_value(y, y1, weight)

        for i in range(2):  # each factor node has only 2 neighbor variable node
            var_neighbor = self.neighbor[i]
            if labeled_given and var_neighbor.label_type == LabelTypeEnum.KNOWN_LABEL:
                msg_y = var_neighbor.y
                msg[i, msg_y] = 1.0
            else:
                msg[i, :] = np.dot(func_values, belief[1-i,:])  # <PRML> p404 Eqn.(8.66)
        msg /= np.sum(msg,axis=1).reshape(-1,1) # normalize, reshape for broadcast in numpy

        var_neighbor.get_message_from(self.id, self.msg, diff_max)  # neighbor[i]是variable_node


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
        self.entry = [] # For each subgraph (connected component), we select one node as entry




    cpdef add_edge(self, int a, int b, FactorFunction func):
        '''
        :param a:  int类型id
        :param b:  int类型id
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
        cdef list filtered_index_list
        cdef np.ndarray[DTYPE_int_t, ndim=1] filtered_index, all_index
        for p_node in self.p_node:
            p_node.add_neighbor_done()

        for var_node in self.var_node:
            filtered_index_list = list()
            all_index = np.arange(len(var_node.neighbor)).astype(np.int32)
            for i in range(len(var_node.neighbor)):
                filtered_index = np.delete(all_index,i,0)
                filtered_index_list.append(filtered_index)
            var_node.filtered_index_list = filtered_index_list

    cpdef clear_data_for_sum_product(self):
        for i in range(self.n):
            self.var_node[i].state_factor[:] = 1.0 / self.num_label
        for i in range(self.num_node):
            for t in range(len(self.p_node[i].neighbor)):
                self.p_node[i].belief[t, :] = 1.0/self.num_label


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
                self.entry.append(self.p_node[i])
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

    cpdef belief_propagation(self, int max_iter, np.ndarray[DTYPE_float_t, ndim=1] weight):
        cdef int start, end, dir,p,iter
        start = 0
        end = 0
        dir = 0
        self.converged = False
        for iter in range(max_iter):
            self.diff_max.diff_max =0.0
            if iter % 2 == 0: # 第偶数次迭代，从后向前传递消息，书上写了，再从根往回传，来回往复，num_node = n + m
                start = self.num_node-1
                end = -1
                dir = -1
            else:
                start = 0
                end = self.num_node
                dir = 1
            for p in range(start, end, dir):
                self.bfs_node[p].belief_propagation(self.diff_max, self.labeled_given, weight) # 这个调用是override动态绑定，根据子类实现的不同调用不同的对象的方法

            if self.diff_max.diff_max < 1e-6:
                break

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
            self.var_node[i].marginal /= sum_py

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
