import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import initializers
from graph_learning.dataset.crf_pact_structure import CRFPackageStructure

class TemporalLSTM(chainer.Chain):

    def __init__(self, box_num, in_size, out_size, use_bi_lstm=True, initialW=None):

        super(TemporalLSTM, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.box_num = box_num

        if not initialW:
            initialW = initializers.HeNormal()
        with self.init_scope():
            self.node_feature_convert_len = 1024
            self.fc1 = L.Linear(in_size, self.node_feature_convert_len, initialW=initialW)
            self.fc2 = L.Linear(self.node_feature_convert_len, self.node_feature_convert_len, initialW=initialW)
            if use_bi_lstm:
                assert out_size % 2 == 0
                for i in range(self.box_num):
                    self.add_link("lstm_{}".format(i),  L.NStepBiLSTM(1, self.node_feature_convert_len,
                                                                    out_size//2, dropout=0.0))
            else:
                for i in range(self.box_num):
                    self.add_link("lstm_{}".format(i), L.NStepLSTM(1, self.node_feature_convert_len,
                                                                     out_size, dropout=0.0))



    def predict(self, x, crf_pact_structure:CRFPackageStructure=None, is_bin=False):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        xp = chainer.cuda.cupy.get_array_module(x)

        with chainer.no_backprop_mode():
            xs = F.expand_dims(x, 0)
            result = self.get_output(xs)
            result = F.copy(result, -1)
            result = F.sigmoid(result)

            pred_score = result.data[0]  # shape = N x out_size
            pred = np.round(pred_score)
            pred = pred.astype(np.int32)    # since each node has multiple labels, we can't use np.argmax here.
            assert len(pred) == x.shape[0]
        return pred.astype(np.int32)  # return N x L, where N is number of nodes. L is label_bin_len


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

    def get_output(self, xs):
        xp = chainer.cuda.get_array_module(xs.data)
        assert xs.shape[0] == 1
        for x in xs:  # xs is shape B x N x D. B is batch_size, always = 1
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            assert x.shape[1] == self.node_feature_convert_len
            x = x.reshape(-1, self.box_num, self.node_feature_convert_len)  # because of padding, T x box_number x D
            x = F.transpose(x, (1, 0, 2))  # shape = box_number x T x D
            x_tuple = F.split_axis(x, self.box_num, axis=0, force_tuple=True)  # list of T x D
            result_list = []  # length = box_num
            for i, x in enumerate(x_tuple):
                lstm = getattr(self, "lstm_{}".format(i))
                _, _, lstm_result = lstm(None, None, [x])
                result_list.append(lstm_result)
            result = F.stack(result_list)  # shape = box_number x T x D
            result = F.transpose(result, (1, 0, 2))  # shape = T x box_number x D
            assert result.shape[2] == self.out_size
            result = result.reshape(-1, self.out_size)  # shape = N x out_size
        # return shape B x N x D. B is batch_size,  but can only deal with one, N is number of variable nodes in graph D is out_size
        return F.expand_dims(result,
                             axis=0)

    def __call__(self, xs, crf_pact_structures):  # xs is chainer.Variable
        '''
        only support batch_size = 1
        some example of NStepLSTM : https://github.com/kei-s/chainer-ptb-nsteplstm/blob/master/train_ptb_nstep.py#L24
        :return : chainer.Variable shape= B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector
        '''

        xp = chainer.cuda.get_array_module(xs.data)
        hs = self.get_output(xs)  # B x N x D, B = 1
        ts = self.get_gt_labels(xp, crf_pact_structures=crf_pact_structures)
        ts = chainer.Variable(ts)
        hs = hs.reshape(-1, hs.shape[-1])
        ts = ts.reshape(-1, ts.shape[-1])
        loss = F.sigmoid_cross_entropy(hs, ts)
        accuracy = F.binary_accuracy(hs,ts)
        chainer.reporter.report({'loss':loss, 'accuracy':accuracy}, self)
        return loss




