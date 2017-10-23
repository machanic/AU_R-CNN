from collections import defaultdict

import chainer
import chainer.functions as F
import numpy as np

import config
from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
from structural_rnn.model.open_crf.cython.factor_graph import FactorGraph
from structural_rnn.model.open_crf.cython.open_crf_layer import OpenCRFLayer
from structural_rnn.model.s_rnn.structural_rnn import StructuralRNN


class StructuralRNNPlus(chainer.Chain):

    # note that out_size must == label_bin_len not combine label_num
    def __init__(self, crf_pact_structure:CRFPackageStructure, in_size, hidden_size, out_size, with_crf=True):
        super(StructuralRNNPlus, self).__init__()
        self.with_crf = with_crf

        if not self.with_crf:
            assert out_size == crf_pact_structure.sample.label_bin_len # num_label指的是二进制label_bin的维度
        else:
            # actually, hidden_size come from args.hidden_size
            assert hidden_size == crf_pact_structure.num_attrib_type  # 由于要用到crf_pact_structure.num_feature, 而setup_graph提前装好的EdgeFunction也要用到这个长度
        self.neg_pos_ratio = 3
        sample = crf_pact_structure.sample
        self.label_dict = sample.label_dict
        n = sample.num_node
        m = sample.num_edge
        num_label = crf_pact_structure.num_label  #  whereas num_label(combine label count) only used in open_crf, this num_label doesn't matter here

        factor_graph = FactorGraph(n=n, m=m, num_label=num_label)  # this factor_graph is only for construct Structural RNN not for OpenCRF derive
        get_frame = lambda e: int(e[0:e.index("_")])
        min_frame = min(map(get_frame, [sample.nodeid_line_no_dict.mapping_dict.inv[node.id] \
                                        for node in sample.node_list]))


        frame_node_id = defaultdict(list)
        remove_var_node_index_ls = list()
        remove_factor_node_index_ls = list()
        for i in range(n):
            nodeid_int = sample.node_list[i].id
            nodeid_str = sample.nodeid_line_no_dict.mapping_dict.inv[nodeid_int]
            frame_node_id[get_frame(nodeid_str)].append(nodeid_int)
            if get_frame(nodeid_str) == min_frame:
                factor_graph.set_variable_label(i, sample.node_list[i].label)
                factor_graph.var_node[i].label_type = sample.node_list[i].label_type
            else:
                remove_var_node_index_ls.append(i)
        for i in range(m):
            frame_a = get_frame(sample.nodeid_line_no_dict.mapping_dict.inv[sample.edge_list[i].a])
            frame_b = get_frame(sample.nodeid_line_no_dict.mapping_dict.inv[sample.edge_list[i].b])
            if frame_a == frame_b == min_frame:  # 只要第一帧的factor_graph的edge，从而不考虑原始的S-RNN论文中自己到自己的边
                factor_graph.add_edge(sample.edge_list[i].a, sample.edge_list[i].b, None)  # func用不到
            else:
                remove_factor_node_index_ls.append(i)
       # We don't need self link EdgeRNN here(in original paper self to self link), I assume this can be done via RNN inherent nature
        for index in sorted(remove_var_node_index_ls, reverse=True):
            del factor_graph.var_node[index]
        for index in sorted(remove_factor_node_index_ls, reverse=True):
            del factor_graph.factor_node[index]
        factor_graph.n = len(factor_graph.var_node)
        factor_graph.m = len(factor_graph.factor_node)
        factor_graph.num_node = factor_graph.n + factor_graph.m

        with self.init_scope():
            if self.with_crf:
                self.structural_rnn = StructuralRNN(factor_graph, in_size, hidden_size)
                self.open_crf = OpenCRFLayer(node_in_size=hidden_size, weight_len=crf_pact_structure.num_feature)
            else:
                self.structural_rnn = StructuralRNN(factor_graph, in_size, out_size)  # 若只有一个模块，则out_size=label_bin_len

    def get_gt_labels(self, xp, crf_pact_structures, is_bin=True):
        targets = list()
        for crf_pact in crf_pact_structures:
            sample = crf_pact.sample
            node_label_one_video = list()
            for node in sample.node_list:
                if is_bin:
                    label_bin = node.label_bin
                    node_label_one_video.append(label_bin)
                else:
                    label = node.label
                    node_label_one_video.append(label)
            targets.append(np.asarray(node_label_one_video))
        return xp.stack(targets)  # shape = B x N x D but B = 1 forever


    def predict(self, x:np.ndarray, crf_pact_structure:CRFPackageStructure):
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
            if self.with_crf:
                xs = F.expand_dims(x,axis=0)
                crf_pact_structures = [crf_pact_structure]
                hs = self.structural_rnn(xs, crf_pact_structures)  # hs shape = B x N x D, B is batch_size
                h = hs.data[0]
                pred_labels = self.open_crf.predict(h, crf_pact_structures[0])  # shape = N x 1
                # this method can only inference label combination that already occur in training dataset
                AU_bins = [] # AU_bins is labels in one video sequence
                for pred_label in pred_labels:  # pred_label is int id of combine AU. multiple AU combine regarded as one
                    AU_list = self.label_dict.get_key(pred_label).split(",") #  actually, because Open-CRF only support single label prediction
                    AU_bin = np.zeros(len(config.AU_SQUEEZE), dtype=np.int32)
                    for AU in AU_list:
                        np.put(AU_bin, config.AU_SQUEEZE.inv[AU], 1)
                    AU_bins.append(AU_bin)
                return xp.asarray(AU_bins)  # shape =N x D, where D = len(config.AU_SQUEEZE) = label_bin_len
            else:
                return self.structural_rnn.predict(x, crf_pact_structure, infered=False)  # 是binary形式的label. N x D



    # return loss
    def __call__(self, xs:chainer.Variable, crf_pact_structures):  # crf_pact_structure is batch of CRFPackageStructure
        xp = chainer.cuda.cupy.get_array_module(xs.data)  # xs is batch
        # return shape = B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector
        h = self.structural_rnn(xs, crf_pact_structures)
        ts = self.get_gt_labels(xp, crf_pact_structures, is_bin=True)  # B x N x D, B = 1 forever
        if self.with_crf:
            #  open_crf only support CPU mode
            h = F.copy(h, -1)
            # gt_label is hidden inside crf_pact_structure's sample. this step directly compute loss
            loss = self.open_crf(h, crf_pact_structures)
        else:  # only s_rnn
            union_gt = set()  # union of prediction positive and ground truth positive
            # ts shape = B * N * D , B is batch_size, N is one video all nodes count, D is each label_bin dimension
            ts = ts.reshape(-1, ts.shape[2])  # shape = (B x N) x D
            h = h.reshape(-1, h.shape[2])  # shape = (B x N) x D
            assert h.shape[-1] == ts.shape[-1]
            # too much negative sample point, we need sub-sampling before compute sigmoid_cross_entropy loss
            # method we first pick gt=1, then false positive where pred=1 but gt=0, if not enough, pick rest
            cpu_ts = chainer.cuda.to_cpu(ts)
            cpu_pred = (chainer.cuda.to_cpu(h.data) > 0).astype(np.int32)
            gt_pos_index = np.nonzero(cpu_ts)
            pred_pos_index = np.nonzero(cpu_pred)
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
                                           ts[list(pick_index[0]), list(pick_index[1])])  # 支持多label

        report_dict = {'loss': loss}
        if not self.with_crf:
            report_dict["accuracy"] = accuracy
        chainer.reporter.report(report_dict,
                                self)
        return loss