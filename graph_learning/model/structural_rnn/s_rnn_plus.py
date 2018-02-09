from collections import defaultdict

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
from graph_learning.model.open_crf.cython.factor_graph import FactorGraph
from graph_learning.model.open_crf.cython.open_crf_layer import OpenCRFLayer
from graph_learning.model.structural_rnn.structural_rnn import StructuralRNN


class StructuralRNNPlus(chainer.Chain):

    # note that out_size must == label_bin_len not combine label_num, TODO 更加合理化：注意这里的out_size 我们传入的是不带0 label_num,所以=label_bin_len
    def __init__(self, crf_pact_structure:CRFPackageStructure, in_size, hidden_size, out_size, with_crf=True,use_bi_lstm=True):
        super(StructuralRNNPlus, self).__init__()
        self.with_crf = with_crf

        if not self.with_crf:
            assert out_size == crf_pact_structure.num_label # num_label指的是二进制label_bin的维度
        else: # if we use crf, hidden_size will be the same of crf_pact_structure.num_attrib_type
            # actually, hidden_size come from args.hidden_size
            assert hidden_size == crf_pact_structure.num_attrib_type, "{0}!={1}".format(hidden_size, crf_pact_structure.num_attrib_type)  # 由于要用到crf_pact_structure.num_feature, 而setup_graph提前装好的EdgeFunction也要用到这个长度
        sample = crf_pact_structure.sample
        n = sample.num_node
        m = sample.num_edge
        num_label = crf_pact_structure.num_label  #  whereas num_label(combine label count) only used in open_crf, this num_label doesn't matter here

        factor_graph = FactorGraph(n=n, m=m, num_label=num_label)  # this factor_graph is only for construct Structural RNN not for OpenCRF derive
        get_frame = lambda e: int(e[0:e.index("_")])
        min_frame = min(map(get_frame, [sample.nodeid_line_no_dict.mapping_dict.inv[node.id] \
                                        for node in sample.node_list]))


        frame_node_id = defaultdict(list)  # frame -> node_id
        remove_var_node_index_ls = list()
        remove_factor_node_index_ls = list()
        for i in range(n):
            node_id = sample.node_list[i].id
            nodeid_str = sample.nodeid_line_no_dict.mapping_dict.inv[node_id]
            frame_node_id[get_frame(nodeid_str)].append(node_id)
            factor_graph.var_node[i].id = node_id
            factor_graph.p_node[node_id] = factor_graph.var_node[i]
            factor_graph.var_node[i].init(crf_pact_structure.num_label)
            factor_graph.set_variable_label(i, sample.node_list[i].label)
            factor_graph.var_node[i].label_type = sample.node_list[i].label_type
            if get_frame(nodeid_str) != min_frame:
                remove_var_node_index_ls.append(i)
        for i in range(m):
            factor_node_id = sample.edge_list[i].id
            factor_graph.factor_node[i].id = factor_node_id
            factor_graph.p_node[factor_node_id] = factor_graph.factor_node[i]
            factor_graph.factor_node[i].init(crf_pact_structure.num_label)
            frame_a = get_frame(sample.nodeid_line_no_dict.mapping_dict.inv[sample.edge_list[i].a])
            frame_b = get_frame(sample.nodeid_line_no_dict.mapping_dict.inv[sample.edge_list[i].b])
            if frame_a == frame_b == min_frame:  # 只要第一帧的factor_graph的edge，从而不考虑原始的S-RNN论文中自己到自己的边
                factor_graph.add_edge(i, sample.edge_list[i].a, sample.edge_list[i].b, sample.edge_list[i].edge_type)  # func用不到
            else:
                remove_factor_node_index_ls.append(i)
       # We don't need self link EdgeRNN here(in original paper self to self link), I assume this can be done via RNN inherent nature
        for index in sorted(remove_var_node_index_ls, reverse=True):
            del factor_graph.var_node[index]  # only preserve the first frame variable node and factor node to construct S-RNN
        for index in sorted(remove_factor_node_index_ls, reverse=True):
            del factor_graph.factor_node[index]
        factor_graph.n = len(factor_graph.var_node)
        factor_graph.m = len(factor_graph.factor_node)
        factor_graph.num_node = factor_graph.n + factor_graph.m

        with self.init_scope():
            if self.with_crf:
                self.structural_rnn = StructuralRNN(factor_graph, in_size, hidden_size,use_bi_lstm=use_bi_lstm)
                self.open_crf = OpenCRFLayer(node_in_size=hidden_size, weight_len=crf_pact_structure.num_feature)
            else:
                self.structural_rnn = StructuralRNN(factor_graph, in_size, out_size, use_bi_lstm=use_bi_lstm)  # 若只有一个模块，则out_size=label_bin_len

    def get_gt_label_one_graph(self, xp, crf_pact_structure, is_bin=True):
        sample = crf_pact_structure.sample
        if not is_bin:
            node_label_one_video = xp.zeros(shape=len(sample.node_list), dtype=xp.int32)
        else:
            node_label_one_video = xp.zeros(shape=(len(sample.node_list), sample.label_bin_len), dtype=xp.int32)
        for idx, node in enumerate(sample.node_list):
            assert node.id == idx
            if is_bin:
                label_bin = node.label_bin
                node_label_one_video[node.id] = label_bin
            else:
                label = node.label
                node_label_one_video[node.id] = label
        return node_label_one_video

    def get_gt_labels(self, xp, crf_pact_structures, is_bin=True):
        targets = list()
        for crf_pact in crf_pact_structures:
            node_label_one_video = self.get_gt_label_one_graph(xp, crf_pact, is_bin)
            targets.append(node_label_one_video)
        return xp.stack(targets).astype(xp.int32)  # shape = B x N x D but B = 1 forever


    def predict(self, x:np.ndarray, crf_pact_structure:CRFPackageStructure,is_bin=False):
        '''
        :param xs:
        :param crf_pact_structures:
        :return: bin array for multi-label, shape= B x N x D
        '''
        with chainer.no_backprop_mode():
            if not isinstance(x, chainer.Variable):
                x = chainer.Variable(x)
            xp = chainer.cuda.get_array_module(x)
            # return shape = B * N * D , B is batch_size(=1 only), N is one video all nodes count, D is each node output vector
            if self.with_crf:  # 作废，这句if不会进去
                xs = F.expand_dims(x, axis=0)
                crf_pact_structures = [crf_pact_structure]
                hs = self.structural_rnn(xs, crf_pact_structures)  # hs shape = B x N x D, B is batch_size
                hs = F.copy(hs, -1) # data transfer to cpu
                h = hs.data[0]
                pred_labels = self.open_crf.predict(h, crf_pact_structures[0],is_bin=is_bin)  # shape = N x D or N x 1
                return np.asarray(pred_labels, dtype=xp.int32)  # shape =N x D, where D = AU_squeeze_size
            else:
                return self.structural_rnn.predict(x, crf_pact_structure, is_bin=is_bin)  # 是binary形式的label. N x D



    # return loss
    def __call__(self, xs:chainer.Variable, crf_pact_structures):  # crf_pact_structure is batch of CRFPackageStructure
        xp = chainer.cuda.cupy.get_array_module(xs.data)  # xs is batch
        # return shape = B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector dimension
        h = self.structural_rnn(xs, crf_pact_structures)

        if self.with_crf:
            #  open_crf only support CPU mode
            # convert_xs = self.bn(self.convert_dim_fc(xs.reshape(-1, xs.shape[-1])))  # note that we remove batch = 1 dimension
            # h = F.relu(h.reshape(-1, h.shape[-1]) + convert_xs) # just like ResNet
            # h = F.expand_dims(h, 0)  # add one batch dimension
            h = F.copy(h, -1)
            # gt_label is hidden inside crf_pact_structure's sample. this step directly compute loss
            loss = self.open_crf(h, crf_pact_structures)
        else:  # only structural_rnn
            ts = self.get_gt_labels(xp, crf_pact_structures, is_bin=False)  # B x N x 1, and B = 1 forever
            ts = chainer.Variable(ts.reshape(-1))  # because ts label is 0~L which is one more than ground truth, 0 represent 0,0,0,0,0
            h = h.reshape(-1, h.shape[-1])  # h must have 0~L which = L+1 including non_AU = 0(also background class)
            assert ts.shape[0] == h.shape[0]
            loss = F.hinge(h, ts, norm='L2', reduce='mean')

            accuracy = F.accuracy(h,ts)

        report_dict = {'loss': loss}
        if not self.with_crf:
            report_dict["accuracy"] = accuracy
        chainer.reporter.report(report_dict,
                                self)
        return loss