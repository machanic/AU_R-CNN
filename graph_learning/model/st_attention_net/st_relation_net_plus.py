from collections import defaultdict

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
from graph_learning.model.open_crf.cython.factor_graph import FactorGraph
from graph_learning.model.open_crf.cython.open_crf_layer import OpenCRFLayer
from graph_learning.model.structural_rnn.structural_rnn import StructuralRNN
import config
from enum import Enum
from itertools import combinations
from graph_learning.model.st_attention_net.st_relation_net import StRelationNet


linear_init = chainer.initializers.LeCunUniform()

class StRelationNetPlus(chainer.Chain):

    # note that out_size must == label_bin_len not combine label_num, TODO 更加合理化：注意这里的out_size 我们传入的是不带0 label_num,所以=label_bin_len
    def __init__(self, crf_pact_structure:CRFPackageStructure, in_size, out_size, database,
                neighbor_mode, spatial_edge_mode, recurrent_block_type,
                 attn_heads:int,dropout:float, use_geometry_features:bool, layers:int, bi_lstm:bool, lstm_first_forward:bool):
        super(StRelationNetPlus, self).__init__()
        self.neg_pos_ratio = 3
        assert out_size == crf_pact_structure.label_bin_len
        self.out_size = out_size
        sample = crf_pact_structure.sample
        n = sample.num_node
        self.frame_node_num = config.BOX_NUM[database]
        assert n % self.frame_node_num == 0

        num_label = crf_pact_structure.num_label  #  whereas num_label(combine label count) only used in open_crf, this num_label doesn't matter here
        factor_graph = FactorGraph(n=n, m=self.frame_node_num * (self.frame_node_num - 1) // 2, num_label=num_label)  # this factor_graph is only for construct Structural RNN not for OpenCRF derive

        get_frame = lambda e: int(e[0:e.index("_")])
        min_frame = min(map(get_frame, [sample.nodeid_line_no_dict.mapping_dict.inv[node.id] \
                                        for node in sample.node_list]))

        remove_var_node_index_ls = list()

        firstframe_node_id_ls = list()
        for i in range(n):
            node_id = sample.node_list[i].id
            nodeid_str = sample.nodeid_line_no_dict.mapping_dict.inv[node_id]
            if get_frame(nodeid_str) == min_frame:
                factor_graph.var_node[i].id = node_id
                assert i == node_id
                firstframe_node_id_ls.append(node_id)
                factor_graph.p_node[node_id] = factor_graph.var_node[i]
                factor_graph.var_node[i].init(crf_pact_structure.num_label)
                factor_graph.set_variable_label(i, sample.node_list[i].label)
                factor_graph.var_node[i].label_type = sample.node_list[i].label_type
            else:
                remove_var_node_index_ls.append(i)
        #for i in range(self.frame_node_num * (self.frame_node_num - 1) // 2):

        for i, (node_a, node_b) in enumerate(combinations(firstframe_node_id_ls, 2)):
            factor_node_id = sample.edge_list[i].id  # note that edge id not start from 0
            factor_graph.factor_node[i].id = factor_node_id
            factor_graph.p_node[factor_node_id] = factor_graph.factor_node[i]
            factor_graph.factor_node[i].init(crf_pact_structure.num_label)
            factor_graph.add_edge(i, node_a, node_b, 0)

        for index in sorted(remove_var_node_index_ls, reverse=True):
            del factor_graph.var_node[index]

        factor_graph.n = len(factor_graph.var_node)
        factor_graph.m = len(factor_graph.factor_node)
        factor_graph.num_node = factor_graph.n + factor_graph.m

        with self.init_scope():
            self.st_relation_net = StRelationNet(factor_graph, n_layers=layers, in_size=in_size,
                                                   out_size=out_size, frame_node_num=self.frame_node_num,
                                                   initialW=linear_init,neighbor_mode=neighbor_mode,
                                                   spatial_edge_model=spatial_edge_mode,
                                                   recurrent_block_type=recurrent_block_type, attn_heads=attn_heads,
                                                   attn_dropout=dropout, use_geometry_features=use_geometry_features,
                                                   bi_lstm=bi_lstm, lstm_first_forward=lstm_first_forward
                                                   )

    def get_gt_label_one_graph(self, xp, crf_pact_structure, is_bin=True, device=-1):
        sample = crf_pact_structure.sample
        if not is_bin:
            node_label_one_video = np.zeros(shape=len(sample.node_list), dtype=np.int32)
        else:
            node_label_one_video = np.zeros(shape=(len(sample.node_list), sample.label_bin_len), dtype=np.int32)
        for idx, node in enumerate(sample.node_list):
            assert node.id == idx
            if is_bin:
                label_bin = node.label_bin
                node_label_one_video[node.id] = label_bin
            else:
                label = node.label
                node_label_one_video[node.id] = label
        if xp!=np:
            node_label_one_video = chainer.cuda.to_gpu(node_label_one_video, device, chainer.cuda.Stream.null)
        return node_label_one_video

    def get_gt_labels(self, xp, crf_pact_structures, is_bin=True, device=-1):
        targets = list()
        for crf_pact in crf_pact_structures:
            node_label_one_video = self.get_gt_label_one_graph(xp, crf_pact, is_bin, device)
            targets.append(node_label_one_video)
        return xp.stack(targets).astype(xp.int32)  # shape = B x N x D but B = 1 forever


    def predict(self, x:np.ndarray,g:np.ndarray, crf_pact_structure:CRFPackageStructure):
        '''
        :param x:
        :param crf_pact_structure:
        :return: bin array for multi-label, shape= B x N x D
        '''
        with chainer.no_backprop_mode():
            if not isinstance(x, chainer.Variable):
                x = chainer.Variable(x)
                g = chainer.Variable(g)
            # return shape = B * N * D , B is batch_size(=1 only), N is one video all nodes count, D is each node output vector
            return self.st_relation_net.predict(x, g, crf_pact_structure)  # 是binary形式的label. N x D

    # return loss
    def __call__(self, xs:chainer.Variable, gs:chainer.Variable, crf_pact_structures):  # crf_pact_structure is batch of CRFPackageStructure
        xp = chainer.cuda.cupy.get_array_module(xs.data)  # xs is batch
        # return shape = B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector dimension
        h = self.st_relation_net(xs, gs, crf_pact_structures)
        ts = self.get_gt_labels(xp, crf_pact_structures, is_bin=True, device=h.data.device)  # B x N x out_size, and B = 1 forever
        batch, N, out_size = ts.shape
        assert out_size == self.out_size
        ts = ts.reshape(-1, ts.shape[-1])  # note that ts is not chainer.Variable
        h = h.reshape(-1, h.shape[-1])  # h must have 0~L which = L+1 including non_AU = 0(also background class)
        assert ts.shape[0] == h.shape[0]

        union_gt = set()  # union of prediction positive and ground truth positive
        cpu_ts = chainer.cuda.to_cpu(ts)
        gt_pos_index = np.nonzero(cpu_ts)
        cpu_pred_score = (chainer.cuda.to_cpu(h.data) > 0).astype(np.int32)
        pred_pos_index = np.nonzero(cpu_pred_score)
        len_gt_pos = len(gt_pos_index[0]) if len(gt_pos_index[0]) > 0 else 1
        neg_pick_count = self.neg_pos_ratio * len_gt_pos
        gt_pos_index_set = set(list(zip(*gt_pos_index)))
        pred_pos_index_set = set(list(zip(*pred_pos_index)))
        union_gt.update(gt_pos_index_set)
        union_gt.update(pred_pos_index_set)
        false_positive_index = np.asarray(list(pred_pos_index_set - gt_pos_index_set))  # shape = n x 2
        gt_pos_index_lst = list(gt_pos_index_set)
        if neg_pick_count <= len(false_positive_index):
            choice_fp = np.random.choice(np.arange(len(false_positive_index)), size=neg_pick_count, replace=False)
            gt_pos_index_lst.extend(list(map(tuple, false_positive_index[choice_fp].tolist())))
        else:
            gt_pos_index_lst.extend(list(map(tuple, false_positive_index.tolist())))
            rest_pick_count = neg_pick_count - len(false_positive_index)
            gt_neg_index = np.where(cpu_ts == 0)
            gt_neg_index_set = set(list(zip(*gt_neg_index)))
            gt_neg_index_set = gt_neg_index_set - set(gt_pos_index_lst)  # remove already picked
            gt_neg_index_array = np.asarray(list(gt_neg_index_set))
            choice_rest = np.random.choice(np.arange(len(gt_neg_index_array)), size=rest_pick_count, replace=False)
            gt_pos_index_lst.extend(list(map(tuple, gt_neg_index_array[choice_rest].tolist())))
        pick_index = list(zip(*gt_pos_index_lst))
        if len(union_gt) == 0:
            accuracy_pick_index = np.where(cpu_ts)
        else:
            accuracy_pick_index = list(zip(*union_gt))

        accuracy = F.binary_accuracy(h[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                     ts[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])
        loss = F.sigmoid_cross_entropy(h[list(pick_index[0]), list(pick_index[1])],
                                       ts[list(pick_index[0]), list(pick_index[1])])

        report_dict = {'loss': loss, 'accuracy':accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss