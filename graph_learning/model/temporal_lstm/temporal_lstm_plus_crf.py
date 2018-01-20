import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import initializers
from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
from graph_learning.model.temporal_lstm.temporal_lstm import TemporalLSTM
from graph_learning.model.open_crf.cython.open_crf_layer import OpenCRFLayer

class TemporalLSTMPlus(chainer.Chain):
    def __init__(self, crf_pact_structure:CRFPackageStructure, sequence_num, in_size, hidden_size, out_size, with_crf=True, use_bi_lstm=True):
        super(TemporalLSTMPlus,self).__init__()
        self.with_crf = with_crf
        if not self.with_crf:
            assert out_size == crf_pact_structure.num_label # num_label指的是二进制label_bin的维度
        else:
            assert hidden_size == crf_pact_structure.num_attrib_type
        with self.init_scope():
            if self.with_crf:
                self.temporal_lstm = TemporalLSTM(sequence_num, in_size, hidden_size, use_bi_lstm)
                self.open_crf = OpenCRFLayer(node_in_size=hidden_size, weight_len=crf_pact_structure.num_feature)
            else:
                self.temporal_lstm = TemporalLSTM(sequence_num, in_size, out_size, use_bi_lstm)
    
    def get_gt_label_one_graph(self, xp, crf_pact_structure, is_bin=True):
        sample = crf_pact_structure.sample
        if not is_bin:
            node_label_one_video = xp.zeros(shape=len(sample.node_list))
        else:
            node_label_one_video = xp.zeros(shape=(len(sample.node_list), sample.label_bin_len))
        for node in sample.node_list:
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

    def predict(self, x: np.ndarray, crf_pact_structure: CRFPackageStructure, is_bin=False):
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
                xs = F.expand_dims(x, axis=0)
                crf_pact_structures = [crf_pact_structure]
                hs = self.temporal_lstm(xs)  # hs shape = B x N x D, B is batch_size
                hs = F.copy(hs, -1)  # data transfer to cpu
                h = hs.data[0]
                pred_labels = self.open_crf.predict(h, crf_pact_structures[0], is_bin=is_bin)  # shape = N x D or N x 1
                return np.asarray(pred_labels, dtype=xp.int32)  # shape =N x D, where D = AU_squeeze_size
            else:
                return self.temporal_lstm.predict(x, crf_pact_structure, is_bin=is_bin)  # 是binary形式的label. N x D


    def __call__(self, xs:chainer.Variable, crf_pact_structures):
        xp = chainer.cuda.cupy.get_array_module(xs.data)
        h = self.temporal_lstm(xs)
        if self.with_crf:
            h = F.copy(h, -1)
            loss = self.open_crf(h, crf_pact_structures)
        else:
            ts = self.get_gt_labels(xp, crf_pact_structures, is_bin=False)  # B x N x 1, and B = 1 forever
            ts = chainer.Variable(ts.reshape(-1))
            h = h.reshape(-1, h.shape[-1])
            assert ts.shape[0] == h.shape[0]
            loss = F.hinge(h, ts, norm='L2', reduce='mean')
            accuracy = F.accuracy(h, ts)
        report_dict = {'loss': loss}
        if not self.with_crf:
            report_dict["accuracy"] = accuracy
        chainer.reporter.report(report_dict,
                                self)
        return loss