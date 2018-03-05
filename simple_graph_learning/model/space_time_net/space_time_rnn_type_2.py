from collections import defaultdict
from itertools import combinations
import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F
import numpy as np
from simple_graph_learning.model.space_time_net.attention_base_block import MultiHeadAttention, \
    PositionwiseFeedForwardLayer, AttentionBlock
import random

from simple_graph_learning.model.space_time_net.enum_type import  RecurrentType, SpatialEdgeMode
import config
from collections import OrderedDict

class TemporalRNN(chainer.Chain):

    def __init__(self, n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(TemporalRNN, self).__init__()
        if not initialW:
            initialW = initializers.HeNormal()
        self.n_layer = n_layers
        self.insize=  insize
        with self.init_scope():
            if use_bi_lstm:
                self.lstm = L.NStepBiLSTM(self.n_layer, 1024, 256,initialW=initialW, dropout=0.1) #dropout = 0.0
            else:
                self.lstm = L.NStepLSTM(self.n_layer, 1024, 512, initialW=initialW,dropout=0.1)
            self.fc1 = L.Linear(insize, 1024, initialW=initialW)
            self.fc2 = L.Linear(1024, 1024, initialW=initialW)
            self.fc3 = L.Linear(512, outsize, initialW=initialW)

    def __call__(self, xs):  # input list of T,D
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hx = None
        cx = None
        xs = F.stack(xs) # batch, T, D
        batch, T, dim = xs.shape
        xs = F.reshape(xs, shape=(-1, self.insize))
        hs = F.relu(self.fc1(xs))
        hs = F.relu(self.fc2(hs))
        hs = F.reshape(hs, shape=(batch, T, -1))
        hs = list(F.separate(hs, axis=0))
        _, _, hs = self.lstm(hx, cx, hs)  # hs is list of T x D variable
        hs = F.stack(hs)
        hs = F.reshape(hs, shape=(-1, 512))
        hs = self.fc3(hs)
        hs = F.reshape(hs, shape=(batch, T, -1))
        hs = list(F.separate(hs, axis=0))
        return hs


class SpaceTimeRNN(chainer.Chain):
    '''
    all combination modes:
       0. use LSTM or AttentionBlock as base module
    edge_RNN module's input:
       1. concatenate of 2 neighbor node features(optional : + geometry features).
       2. 'none_edge': there is no edge_RNN, so input of node_RNN became "object relation features"
    node_RNN module's input:
       1. use LSTM or AttentionBlock as base module
       2. 'concat' : concatenate of all neighbor edge_RNN output, shape = ((neighbor_edge + 1), converted_dim)
       3. 'none_edge' & 'attention_fuse': use object relation module to obtain weighted sum of neighbor \
                                                 node appearance feature (optional : + geometry features).
       4. 'none_edge' & 'no_neighbor': do not use edge_RNN and just use node appearance feature itself input to node_RNN
    '''

    def __init__(self, database, n_layers:int, in_size:int, out_size:int, initialW=None,
                 spatial_edge_model: SpatialEdgeMode = SpatialEdgeMode.all_edge,
                 recurrent_block_type: RecurrentType = RecurrentType.rnn, attn_heads=8, bi_lstm=False):
        super(SpaceTimeRNN, self).__init__()
        self.neg_pos_ratio = 3
        self.database = database
        self.spatial_edge_mode = spatial_edge_model
        self.out_size = out_size
        self.in_size = in_size
        self.frame_node_num = config.BOX_NUM[self.database]
        self.mid_size = 1024
        NodeRecurrentModule = AttentionBlock if recurrent_block_type == RecurrentType.attention_block else TemporalRNN
        if recurrent_block_type == RecurrentType.no_temporal:
            NodeRecurrentModule = PositionwiseFeedForwardLayer

        with self.init_scope():
            if not initialW:
                initialW = initializers.HeNormal()

            self.top = dict()
            for i in range(self.frame_node_num):
                if recurrent_block_type == RecurrentType.rnn:
                    self.add_link("Node_{}".format(i),
                                  TemporalRNN(n_layers, self.in_size, self.mid_size, use_bi_lstm=bi_lstm))
                else:
                    self.add_link("Node_{}".format(i),
                                  NodeRecurrentModule(n_layers, self.in_size, self.mid_size))
                self.top[str(i)] = getattr(self, "Node_{}".format(i))

            fc_in_len = self.mid_size
            if spatial_edge_model != SpatialEdgeMode.no_edge:
                self.space_lstm = L.NStepBiLSTM(n_layers, self.in_size, self.mid_size//2, dropout=0.1, initialW=initialW)
                fc_in_len = self.mid_size * 2

            self.fc = L.Linear(fc_in_len, self.out_size, initialW=initialW)



    def node_recurrent_forward(self,xs): # xs is shape of (batch, T,F,D)
        node_out_dict = OrderedDict()
        for node_module_id, node_module in self.top.items():
            input_x = xs[:, :, int(node_module_id), :]  # B, T, D
            input_x = F.split_axis(input_x, input_x.shape[0], axis=0, force_tuple=True)
            input_x = [F.squeeze(x) for x in input_x]  # list of T,D
            node_out_dict[node_module_id] = F.stack(node_module(input_x)) # B, T, out_size
        return node_out_dict



    def forward(self, xs):  # xs shape = (batch, T, F, D)
        '''
        :param xs: appearance features of all boxes feature across all frames
        :param gs:  geometry features of all polygons. each is 4 coordinates represent box
        :param crf_pact_structures: packaged graph structure contains supplementary information
        :return:
        '''
        xp = chainer.cuda.get_array_module(xs.data)
        batch = xs.shape[0]
        T = xs.shape[1]
        dim = xs.shape[-1]
        # first frame node_id ==> other frame node_id in same corresponding box
        if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
            input_space = F.reshape(xs, shape=(-1, self.frame_node_num, dim)) # batch x T, F, D
            # input_space = F.reshape(xs, (batch * T, self.frame_node_num, dim))  # batch, T*F, D
            input_space = F.separate(input_space, axis=0) # fusing temporal information of batch , each is T*F, D

            _, _, space_out = self.space_lstm(None, None, list(input_space))
            space_out = F.stack(space_out)  # batch*T, F, D
            space_out = F.reshape(space_out, (-1, self.mid_size)) # batch * T * F, D

        temporal_in = xs
        node_out_dict = self.node_recurrent_forward(temporal_in)
        # shape = F, B, T, mid_size
        node_out = F.stack([node_out_ for _, node_out_ in sorted(node_out_dict.items(),
                                                                 key=lambda e: int(e[0]))])
        node_out = F.transpose(node_out, (1,2,0,3))  # shape = (B,T,F,D)

        node_out = F.reshape(node_out, (-1, self.mid_size))
        concat_out = node_out
        if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
            concat_out = F.concat((node_out, space_out), axis=-1)  # shape= B*T*F, 2D
        concat_out = self.fc(concat_out)
        concat_out = F.reshape(concat_out, (batch, T, self.frame_node_num, self.out_size))
        # assert self.frame_node_num == node_out.shape[2],node_out.shape[2]
        # assert self.out_size == node_out.shape[-1]
        # assert T == node_out.shape[1]
        return concat_out


    def get_loss_index(self, pred, ts):
        union_gt = set()  # union of prediction positive and ground truth positive
        cpu_ts = chainer.cuda.to_cpu(ts)
        gt_pos_index = np.nonzero(cpu_ts)
        cpu_pred_score = (chainer.cuda.to_cpu(pred.data) > 0).astype(np.int32)
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
            choice_rest = np.random.choice(np.arange(len(gt_neg_index_array)), size=rest_pick_count, replace=True)
            gt_pos_index_lst.extend(list(map(tuple, gt_neg_index_array[choice_rest].tolist())))
        pick_index = list(zip(*gt_pos_index_lst))
        if len(union_gt) == 0:
            accuracy_pick_index = np.where(cpu_ts)
        else:
            accuracy_pick_index = list(zip(*union_gt))
        return pick_index, accuracy_pick_index


    def __call__(self, xs, bboxes, labels):  # all shape is (B, T, F, D)

        node_out = self.forward(xs)  # node_out B,T,F,D
        node_out = F.reshape(node_out, (-1, self.out_size))
        node_labels = self.xp.reshape(labels, (-1, self.out_size))
        pick_index, accuracy_pick_index = self.get_loss_index(node_out, node_labels)
        loss = F.sigmoid_cross_entropy(node_out,
                                        node_labels)
        accuracy = F.binary_accuracy(node_out[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                     node_labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])

        report_dict = {'loss': loss, "accuracy":accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss

    # can only predict one frame based on previous T-1 frame feature
    def predict(self, xs): # all shape is (B, T, F, D), but will only predict last frame output
        if not isinstance(xs, chainer.Variable):
            xs = chainer.Variable(xs)
        xp = chainer.cuda.cupy.get_array_module(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False) :
            node_out = self.forward(xs) # node_out B,T,F,D
            node_out = chainer.cuda.to_cpu(node_out.data)
            node_out = node_out[:, -1, :, :] # B, F, D
            pred = (node_out > 0).astype(np.int32)
            pred = np.bitwise_or.reduce(pred,axis=1)  # B, D

        return pred  # return batch x out_size, it is last time_step frame of 2-nd axis of input xs prediction

