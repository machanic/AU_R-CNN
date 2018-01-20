import math
from abc import ABCMeta, abstractmethod

import numpy as np
from overrides import overrides

from graph_learning.model.open_crf.pure_python.constant_variable import LabelTypeEnum, DiffMax

var_belief_call_num = 0

class FactorFunction(metaclass=ABCMeta):

    @abstractmethod
    def get_value(self,y1,y2, weight:np.ndarray):
        pass


class EdgeFactorFunction(FactorFunction):

    def __init__(self, num_label, edge_type:int,
                 num_edge_feature_each_type:int, num_attrib_parameter:int, edge_feature_offset:dict):
        super(EdgeFactorFunction, self).__init__()
        self.num_label = num_label
        # self.weight = p_lambda  # 这个p_lambda原来是偏移过的指针,FIXME，这句话要特别注意修改好
        self.feature_offset = edge_feature_offset
        self.edge_type = edge_type
        self.num_edge_feature_each_type = num_edge_feature_each_type
        self.num_attrib_parameter = num_attrib_parameter

    @overrides
    def get_value(self,y1:int, y2:int, weight:np.ndarray):
        a = y1 if y1 < y2 else y2
        b = y1 if y1 > y2 else y2
        i = self.feature_offset[a * self.num_label + b]
        value = weight[self.num_attrib_parameter + self.edge_type * self.num_edge_feature_each_type + i]

        return math.exp(value)


class Node(metaclass=ABCMeta):

    def __init__(self):
        self.id = None
        self.num_label = None
        self.neighbor = list()  # vector<Node*> neighbor
        self.neighbor_pos = dict() # neighbor => position (index in neighbor vector)
        self.msg = None # length == num_label
        self.belief = [] # 这就是所谓信念，其实是二维数组，第一维是邻居个数，第二维是label_num（L）。邻居数量不定，为何第一个维度是vector原因也在于此
        self.feature_value = []  # 不论edge或node都有feature_value

    @abstractmethod
    def init(self, num_label):
        pass

    def basic_init(self, num_label):
        self.num_label = num_label
        self.msg = np.zeros(num_label).astype(np.float32)

    def __del__(self):
        for arr in self.belief:
            del arr
        if self.msg:
            self.msg = None

    def add_neighbor(self, ng: 'Node'):
        self.neighbor_pos[ng.id] = len(self.neighbor)
        self.neighbor.append(ng)
        self.belief.append(np.zeros(self.num_label, dtype=np.float32))

    def normalize_message(self):
        s = 0.0
        for y in range(self.num_label):
            s += self.msg[y]
        for y in range(self.num_label):
            self.msg[y] /= s

    @abstractmethod
    def belief_propagation(self,diff_max:DiffMax, labeled_given:bool, weight:np.ndarray):
        pass

    @abstractmethod
    def max_sum_propagation(self,diff_max:DiffMax, labeled_given:bool, weight:np.ndarray):
        pass


    def get_message_from(self, u:int, msg_vec:np.ndarray, diff_max:DiffMax):
        p = self.neighbor_pos[u]
        for y in range(self.num_label):
            if abs(self.belief[p][y] - msg_vec[y]) > diff_max.diff_max:
                diff_max.diff_max = abs(self.belief[p][y] - msg_vec[y])
            self.belief[p][y] = msg_vec[y]  #哈哈，为他的邻居设置belief为msg，不管该邻居是factor还是variable

class VariableNode(Node):
    def __init__(self):
        super(VariableNode, self).__init__()
        self.y = None # label, may be np.array or true AU
        self.label_type = None
        self.state_factor = None  # 长度是y，就是label个数的数组
        self.marginal = None  # variable node的marginal是长度为num_label的一维数组，而FactorNode的marginal是num_label x num_label的二维数组

    @overrides
    def init(self, num_label):
        super(VariableNode, self).basic_init(num_label)
        self.state_factor = np.zeros(num_label, dtype=np.float32)
        self.marginal = np.zeros(num_label, dtype=np.float32)

    @overrides
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
                        product *= self.belief[j][y]  # 《PRML》 p406的公式(8.69)，因为一个factor node只有两个variable node邻居，去掉公式中连乘符号，公式里所谓的x其实就是label
                self.msg[y] = product
            self.normalize_message()
            f.get_message_from(self.id, self.msg, diff_max)  # f就是第i个neighbor，是factor_node

    @overrides
    def max_sum_propagation(self,diff_max:DiffMax, labeled_given:bool, weight:np.ndarray):
        for i in range(len(self.neighbor)):
            f = self.neighbor[i]
            for y in range(self.num_label):
                product = self.state_factor[y]
                for j in range(len(self.neighbor)):
                    if i != j:
                        product *= self.belief[j][y]  # MaxSum在这个地方还是用乘法，因为没有做ln处理，与PRML的p413不一致（那里是加法）：《PRML》p413 （8.94）
                self.msg[y] = product

            self.normalize_message()
            f.get_message_from(self.id, self.msg, diff_max)  # neighbor[i]是factor_node

    def __del__(self):
        if self.state_factor is not None:
            self.state_factor = None
        if self.marginal is not None:
            self.marginal = None


class FactorNode(Node):

    def __init__(self):
        super(FactorNode, self).__init__()
        self.edge_type = None
        self.marginal = None  # FactorNode's marginal is num_label x num_label 2-D array
        self.func = None  # this func is EdgeFunction which is determined by marginal

    @overrides
    def init(self, num_label):
        super(FactorNode, self).basic_init(num_label)
        self.marginal = np.zeros(shape=(num_label, num_label)).astype(np.float32)

    @overrides
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

    @overrides
    def max_sum_propagation(self,diff_max:DiffMax, labeled_given:bool, weight:np.ndarray):
        for i in range(2):
            if labeled_given and self.neighbor[i].label_type == LabelTypeEnum.KNOWN_LABEL:
                self.msg[:] = 0
                self.msg[self.neighbor[i].y] = 1.0
            else:
                for y in range(self.num_label):
                    s = self.func.get_value(y, 0, weight) * self.belief[1-i][0]
                    for y1 in range(self.num_label):
                        tmp = self.func.get_value(y, y1, weight) * self.belief[1-i][y1]
                        if tmp > s:
                            s = tmp # find max 《PRML》 p413 formular:（8.93）
                    self.msg[y] = s
                self.normalize_message()
            self.neighbor[i].get_message_from(self.id, self.msg, diff_max)

    def __del__(self):
        if self.marginal is not None:
            self.marginal = None
            del self.marginal


class FactorGraph(object):
    # this is original InitGraph code
    def __init__(self, n:int, m:int, num_label:int, func_list:list):
        self.labeled_given = False
        self.n = n
        self.m = m
        self.num_label = num_label
        self.num_node = self.n + self.m
        self.var_node = [VariableNode() for _ in range(self.n)]  #  本来代码是n个长度的var_node数组， key is node_id, which is frame_boxid
        self.factor_node = [FactorNode() for _ in range(self.m)]
        self.bfs_node = np.empty(self.num_node, dtype=Node)
        self.p_node = np.empty(self.num_node, dtype=Node) # p_node contains all node
        self.edge_type_func_list = func_list

        self.diff_max = DiffMax(diff_max=0.0)
        self.converged = False
        self.entry = [] # For each subgraph (connected component), we select one node as entry

    def add_edge(self, factor_node_index:int, a:int, b:int,  edge_type:int):
        '''
        :param a:  int类型id
        :param b:  int类型id
        :param edge_type: the difference of factor_node.func is in edge_type
        :return:
        '''
        self.factor_node[factor_node_index].edge_type = edge_type
        self.factor_node[factor_node_index].add_neighbor(self.var_node[a])
        self.factor_node[factor_node_index].add_neighbor(self.var_node[b])
        self.var_node[a].add_neighbor(self.factor_node[factor_node_index])
        self.var_node[b].add_neighbor(self.factor_node[factor_node_index])


    def clear_data_for_sum_product(self):
        for i in range(self.n):
            self.var_node[i].state_factor[:] = 1.0 / self.num_label
        for i in range(self.num_node):
            for t in range(len(self.p_node[i].neighbor)):
                self.p_node[i].belief[t][:] = 1.0/self.num_label


    def set_variable_label(self, u:int, y:int):
        self.var_node[u].y = y

    def set_variable_state_factor(self, u:int, y:int, v:float):
        self.var_node[u].state_factor[y] = v

    def gen_propagate_order(self):
        mark = np.full(shape=self.num_node, fill_value=False, dtype=bool) # visit mark
        self.bfs_node = np.empty(self.num_node, dtype=Node)
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

    def belief_propagation(self, max_iter, weight):

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

            # if self.diff_max.diff_max < 1e-6:
            #     print("break: {}".format(self.diff_max.diff_max))
            #     break



    def calculate_marginal(self, weight):
        for i in range(self.n):
            sum_py = 0.0
            for y in range(self.num_label):
                self.var_node[i].marginal[y] = self.var_node[i].state_factor[y]
                for t in range(len(self.var_node[i].neighbor)):
                    self.var_node[i].marginal[y] *= self.var_node[i].belief[t][y]
                sum_py += self.var_node[i].marginal[y]
            for y in range(self.num_label):
                self.var_node[i].marginal[y] /= sum_py

        for i in range(self.m):
            sump = 0.0
            for a in range(self.num_label):
                for b in range(self.num_label):
                    self.factor_node[i].marginal[a][b] += \
                        self.factor_node[i].belief[0][a] \
                        * self.factor_node[i].belief[1][b] \
                        * self.factor_node[i].func.get_value(a, b, weight)
                    sump += self.factor_node[i].marginal[a][b]
            for a in range(self.num_label):
                for b in range(self.num_label):
                    self.factor_node[i].marginal[a][b] /= sump

    def max_sum_propagation(self, max_iter:int, weight:np.ndarray):
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

    def clean(self):
        if self.var_node:
            self.var_node.clear()
        if self.factor_node:
            self.factor_node.clear()
        self.bfs_node = None
        self.p_node = None

    def __del__(self):
        self.clean()
