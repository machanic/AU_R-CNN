import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import initializers


class TemporalLSTM(chainer.Chain):

    def __init__(self, sequence_num, in_size, out_size, use_bi_lstm=True, initialW=None):

        super(TemporalLSTM, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.sequence_num = sequence_num
        if not initialW:
            initialW = initializers.HeNormal()
        with self.init_scope():
            self.node_feature_convert_len = 1024
            self.fc1 = L.Linear(in_size, self.node_feature_convert_len, initialW=initialW)
            self.fc2 = L.Linear(self.node_feature_convert_len, self.node_feature_convert_len, initialW=initialW)
            if use_bi_lstm:
                assert out_size % 2 ==0
                self.lstm = L.NStepBiLSTM(1, self.node_feature_convert_len, out_size//2, dropout=0.0)
            else:
                self.lstm = L.NStepLSTM(1, self.node_feature_convert_len, out_size, dropout=0.0)


    def predict(self, x, crf_pact_structure=None, is_bin=False):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        xp = chainer.cuda.cupy.get_array_module(x)

        with chainer.no_backprop_mode():
            xs = F.expand_dims(x, 0)
            xs = self.__call__(xs)
            xs = F.copy(xs, -1)
            pred_score_array = xs.data[0]
            pred = np.argmax(pred_score_array, axis=1)
            assert len(pred) == x.shape[0]
        return pred.astype(np.int32)  # return N x 1, where N is number of nodes. note that we predict label = 0...L

    def __call__(self, xs):  # xs is chainer.Variable
        '''
        only support batch_size = 1
        some example of NStepLSTM : https://github.com/kei-s/chainer-ptb-nsteplstm/blob/master/train_ptb_nstep.py#L24
        :return : chainer.Variable shape= B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector
        '''

        xp = chainer.cuda.get_array_module(xs.data)
        for x in xs:  # xs is shape B x N x D. B is batch_size, always = 1
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            assert x.shape[1]==self.node_feature_convert_len
            x = x.reshape(-1, self.sequence_num, self.node_feature_convert_len)  # because of padding, T x box_number x D
            x = F.transpose(x, (1,0,2)) # shape = box_number x T x D
            x_lst = [x_i for x_i in x] # list of T x D
            _, _, result = self.lstm(None, None, x_lst)
            result = F.stack(result) # shape = box_number x T x D
            result = F.transpose(result, (1,0,2)) # shape = T x box_number x D
            result = result.reshape(-1, self.out_size) # shape = N x out_size
        return F.expand_dims(result,
                             axis=0)  # return shape B x N x D. B is batch_size,  but can only deal with one, N is number of variable nodes in graph D is out_size


