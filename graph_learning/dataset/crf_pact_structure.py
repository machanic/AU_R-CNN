from collections import defaultdict

import config
from graph_learning.dataset.graph_dataset_reader import GlobalDataSet, DataSample

if config.OPEN_CRF_CONFIG["use_pure_python"]:
    from graph_learning.model.open_crf.pure_python.factor_graph import EdgeFactorFunction, FactorGraph
else:
    from graph_learning.model.open_crf.cython.factor_graph import EdgeFactorFunction, FactorGraph
import numpy as np

# this call contains structural of spatial-temporal graph, wich will be use by s-rnnn and open-crf
# this class mean to be composite of `factor_graph` and `DataSample` class which used to pass to open_crf_layer
# this class should only contains one graph_backup or one part of graph_backup which will then pass to open_crf_layer __call__ method
class CRFPackageStructure(object):

    def __init__(self, sample: DataSample, train_data: GlobalDataSet, num_attrib=None, need_s_rnn=True,
                 need_adjacency_matrix=False, need_factor_graph=True):

        self.sample = sample
        self.num_node = self.sample.num_node
        self.num_label = train_data.num_label
        self.label_bin_len = train_data.label_bin_len
        if num_attrib is not None:
            self.num_attrib_type = num_attrib
        else:
            self.num_attrib_type = train_data.num_attrib_type
        self.num_edge_type = train_data.num_edge_type
        self.edge_feature_offset = dict()
        self.num_edge_feature_each_type = 0
        self.num_attrib_parameter = 0
        self.num_feature = self.gen_feature()
        if need_factor_graph:
            self.factor_graph = self.setup_factor_graph()
        self.max_bp_iter = config.OPEN_CRF_CONFIG["max_bp_iter"]
        self.node_id_convert = dict()
        if need_adjacency_matrix:
            self.A = self.setup_adjacency_matrix()
        if need_s_rnn:
            get_frame = lambda e: int(e[0:e.index("_")])
            get_box = lambda e: int(e[e.index("_")+1:])
            box_min_id = dict()
            for node in self.sample.node_list:
                node_key_str = self.sample.nodeid_line_no_dict.mapping_dict.inv[node.id] # key: node_id, val: line number
                box_id = get_box(node_key_str)  # each node_id in graph file consists of "frame_boxid"
                if box_id not in box_min_id:
                    box_min_id[box_id] = node
                elif get_frame(self.sample.nodeid_line_no_dict.mapping_dict.inv[box_min_id[box_id].id]) \
                        > get_frame(self.sample.nodeid_line_no_dict.mapping_dict.inv[node.id]) :
                    box_min_id[box_id] = node
            for node in self.sample.node_list:
                node_key_str = self.sample.nodeid_line_no_dict.mapping_dict.inv[node.id]
                box_id = get_box(node_key_str)
                self.node_id_convert[node.id] = box_min_id[box_id].id   # node_id_convert: all the node transform to first frame correspond box


            self.nodeRNN_id_dict = defaultdict(list)
            for node_id, nodeRNN_id in sorted(self.node_id_convert.items(), key=lambda e: int(e[0])):
                self.nodeRNN_id_dict[nodeRNN_id].append(node_id)

    def setup_adjacency_matrix(self):
        A = np.zeros(shape=(self.num_node, self.num_node), dtype=np.int32)
        for edge in self.sample.edge_list:
            A[edge.a, edge.b] = 1
            A[edge.b, edge.a] = 1
        np.fill_diagonal(A, 1)
        return A


    # this function is used by open-crf layer
    def gen_feature(self):
        num_feature = 0
        self.num_attrib_parameter = self.num_label * self.num_attrib_type  # feature有多少种 x num_label
        num_feature += self.num_attrib_parameter
        self.edge_feature_offset.clear()
        offset = 0
        for y1 in range(self.num_label):
            for y2 in range(y1, self.num_label):
                self.edge_feature_offset[y1 * self.num_label + y2] = offset
                offset += 1
        self.num_edge_feature_each_type = offset
        num_feature += self.num_edge_type * self.num_edge_feature_each_type
        return num_feature

    def setup_factor_graph(self):  # must called after gen_feature
        func_list = []
        for i in range(self.num_edge_type):
            func_list.append(EdgeFactorFunction(num_label=self.num_label, edge_type=i,
                                                num_edge_feature_each_type=self.num_edge_feature_each_type,
                                                num_attrib_parameter=self.num_attrib_parameter,
                                                edge_feature_offset=self.edge_feature_offset))
        n = self.sample.num_node
        m = self.sample.num_edge
        factor_graph = FactorGraph(n=n, m=m, num_label=self.num_label,func_list=func_list)


        for i in range(n):  # add node info
            node_id = self.sample.node_list[i].id
            factor_graph.var_node[i].id = node_id
            factor_graph.p_node[node_id] = factor_graph.var_node[i]
            factor_graph.var_node[i].init(self.num_label)  # init marginal
            factor_graph.set_variable_label(i, self.sample.node_list[i].label)  # 这个label是int类型，代表字典里的数字
            factor_graph.var_node[i].label_type = self.sample.node_list[i].label_type  # ENUM的 KNOWN 或者UNKOWN

        for i in range(m):  # add edge info, mandatory. 注意a和b是int的类型的node-id
            factor_node_id = self.sample.edge_list[i].id
            factor_graph.factor_node[i].id = factor_node_id
            factor_graph.p_node[factor_node_id] = factor_graph.factor_node[i]
            factor_graph.factor_node[i].init(self.num_label)  # init marginal
            factor_graph.add_edge(i, self.sample.edge_list[i].a, self.sample.edge_list[i].b,
                                  self.sample.edge_list[i].edge_type)

        if hasattr(factor_graph,"add_edge_done"):
            factor_graph.add_edge_done()
        factor_graph.gen_propagate_order()
        return factor_graph