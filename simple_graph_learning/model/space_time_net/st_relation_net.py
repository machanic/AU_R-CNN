from collections import defaultdict
import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F
import numpy as np
from enum import Enum
from graph_learning.model.st_attention_net.attention_base_block import MultiHeadAttention, \
    PositionwiseFeedForwardLayer, AttentionBlock
from graph_learning.model.st_attention_net.enum_type import RecurrentType, NeighborMode, SpatialEdgeMode
from functools import partial
from graph_learning.model.open_crf.cython.factor_graph import FactorGraph
import random
from itertools import permutations

class SpatialTemporalRNN(chainer.Chain):

    def __init__(self, n_layers, insize, outsize, relation_module_num, attn_dropout=0.1, n_head=16, initialW=None,
                 use_bi_lstm=False, lstm_first_forward=True, recurrent_type:RecurrentType=RecurrentType.rnn,
                 neighbor_mode:NeighborMode=NeighborMode.attention_fuse):

        super(SpatialTemporalRNN, self).__init__()
        assert neighbor_mode in [NeighborMode.attention_fuse, NeighborMode.no_neighbor]
        if not initialW:
            initialW = initializers.LeCunNormal()
        self.n_layer = n_layers
        self.out_size = outsize
        self.lstm_first_forward = lstm_first_forward
        if recurrent_type != RecurrentType.rnn:
            assert use_bi_lstm == False
        if use_bi_lstm:
            assert outsize % 2 ==0
        with self.init_scope():
            lstm_in_size = insize if lstm_first_forward else 1024
            if recurrent_type == RecurrentType.rnn:
                if use_bi_lstm:
                    self.lstm1 = L.NStepBiLSTM(self.n_layer, lstm_in_size, 512, initialW=initialW, dropout=0.1)
                    self.lstm = partial(self.lstm1,
                                         None,None)#dropout = 0.0
                else:
                    self.lstm1 = L.NStepLSTM(self.n_layer, lstm_in_size, 1024, initialW=initialW, dropout=0.1)
                    self.lstm = partial(self.lstm1,
                                         None,None)
            elif recurrent_type == RecurrentType.attention_block:
                self.lstm1 = AttentionBlock(self.n_layer, lstm_in_size, 1024, max_length=10000)
                self.lstm = self.lstm1
            elif recurrent_type == RecurrentType.no_temporal:
                self.lstm1 = PositionwiseFeedForwardLayer(self.n_layer, lstm_in_size, 1024)
                self.lstm = self.lstm1
            if self.lstm_first_forward:
                self.fc2 = L.Linear(1024, 1024, initialW=initialW)
            else:
                self.fc2 = L.Linear(insize, 1024, initialW=initialW)
            self.fc3 = L.Linear(1024, 1024, initialW=initialW)
            self.fc4 = L.Linear(1024, outsize, initialW=initialW)
            self.relation_module_name = []
            if neighbor_mode == NeighborMode.attention_fuse:
                for i in range(1, relation_module_num+1):
                    name = "multi_head_spatial_attn_{}".format(i)
                    d_k = 1024//n_head
                    d_v = 1024//n_head
                    self.add_link(name, MultiHeadAttention(n_head, d_model=1024, d_k=d_k,d_v=d_v,dropout=attn_dropout))
                    self.relation_module_name.append(name)

    def lstm_first_forward_func(self,xs):
        # xs T,F,in_size
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hx = None
        cx = None
        xs = F.transpose(xs, axes=(1, 0, 2))  # shape = F,T,in_size
        xs = [F.squeeze(e) for e in F.split_axis(xs, xs.shape[0], axis=0, force_tuple=True)]
        _, _, hs = self.lstm(xs)  # hs is list of T x D variable
        hs = F.stack(hs)
        box_num, frame, _ = hs.shape
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        hs = F.relu(self.fc2(hs))
        hs = F.reshape(hs, shape=(box_num, frame, -1))
        hs = F.transpose(hs, axes=(1, 0, 2))  # shape = T, F, 1024
        for relation_module_str in self.relation_module_name[:len(self.relation_module_name) // 2]:
            hs = getattr(self, relation_module_str)(hs, hs, hs)  # shape = T,F, 1024
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        hs = F.relu(self.fc3(hs))
        hs = F.reshape(hs, shape=(frame, box_num, -1))
        for relation_module_str in self.relation_module_name[len(self.relation_module_name) // 2:]:
            hs = getattr(self, relation_module_str)(hs, hs, hs)  # shape = T,F, 1024
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        hs = self.fc4(hs)

        hs = F.reshape(hs, shape=(frame, box_num, self.out_size))
        return hs

    def lstm_last_forward_func(self, xs):
        # xs T,F,in_size
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        frame, box_num, _ = xs.shape
        hs = F.reshape(xs, (-1, xs.shape[-1]))
        hs = F.relu(self.fc2(hs))
        hs = F.reshape(hs, shape=(frame, box_num, -1))
        for relation_module_str in self.relation_module_name[:len(self.relation_module_name) // 2]:
            hs = getattr(self, relation_module_str)(hs, hs, hs)  # shape = T,F, 1024
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        hs = F.relu(self.fc3(hs))
        hs = F.reshape(hs, shape=(frame, box_num, -1))
        for relation_module_str in self.relation_module_name[len(self.relation_module_name) // 2:]:
            hs = getattr(self, relation_module_str)(hs, hs, hs)  # shape = T,F, 1024


        xs = F.transpose(hs, axes=(1, 0, 2))  # shape = F,T,in_size
        xs = [F.squeeze(e) for e in F.split_axis(xs, xs.shape[0], axis=0, force_tuple=True)]
        _, _, hs = self.lstm(xs)  # hs is list of T x D variable
        hs = F.stack(hs)  # F,T,1024
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        hs = self.fc4(hs)
        hs = F.reshape(hs, (box_num, frame, self.out_size))
        hs = F.transpose(hs, (1,0,2))
        return hs

    def __call__(self, xs):
        if self.lstm_first_forward:
            return self.lstm_first_forward_func(xs)
        return self.lstm_last_forward_func(xs)

class StRelationNet(chainer.Chain):
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

    def __init__(self, G:FactorGraph, n_layers:int, in_size:int, out_size:int, frame_node_num:int, initialW=None,
                 neighbor_mode:NeighborMode=NeighborMode.attention_fuse, spatial_edge_model:SpatialEdgeMode=SpatialEdgeMode.all_edge,
                 recurrent_block_type: RecurrentType = RecurrentType.rnn, attn_heads=16, attn_dropout=0.1,
                 use_geometry_features=True, bi_lstm=False, lstm_first_forward=True):
        super(StRelationNet, self).__init__()
        self.spatial_edge_mode = spatial_edge_model
        self.use_geometry_features = use_geometry_features
        self.neighbor_mode = neighbor_mode
        self.out_size = out_size
        self.in_size = in_size
        self.frame_node_num = frame_node_num
        self.mid_size = 1024
        self.attn_dropout = attn_dropout
        self.batch = 1
        assert neighbor_mode in [NeighborMode.attention_fuse, NeighborMode.no_neighbor]


        with self.init_scope():
            if not initialW:
                initialW = initializers.LeCunNormal()
                # we want to use node feature to attend all neighbor feature
            self.head = attn_heads
            self.st_module = SpatialTemporalRNN(n_layers, in_size, out_size, 4, attn_dropout, attn_heads,
                                                initialW=initialW,
                                                use_bi_lstm=bi_lstm, lstm_first_forward=lstm_first_forward,
                                                recurrent_type=recurrent_block_type, neighbor_mode=neighbor_mode)


    def forward(self, xs, gs, crf_pact_structures):  # xs shape = (batch, N, D), batch forever = 1
        '''
        :param xs: appearance features of all boxes feature across all frames
        :param gs:  geometry features of all polygons. each is 4 coordinates represent box
        :param crf_pact_structures: packaged graph structure contains supplementary information
        :return:
        '''
        xp = chainer.cuda.get_array_module(xs.data)

        assert xs.shape[0] == self.batch
        T = xs.shape[1] // self.frame_node_num
        xs = xs.reshape(T, self.frame_node_num, self.in_size) # shape = (batch * T * box_num, D)
        hs = self.st_module(xs)
        hs = F.reshape(hs, shape=(self.batch, -1, self.out_size))
        return hs



    def __call__(self, xs, gs, crf_pact_structures):
        return self.forward(xs, gs, crf_pact_structures)

    def predict(self, x, g, crf_pact_structure):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        xp = chainer.cuda.cupy.get_array_module(x)

        with chainer.no_backprop_mode():
            xs = F.expand_dims(x, 0)
            gs = F.expand_dims(g, 0)
            crf_pact_structures = [crf_pact_structure]
            xs = self.forward(xs, gs, crf_pact_structures)
            pred_score_array = F.copy(xs, -1)  # shape = B,N,D
            pred_score_array = pred_score_array.data[0] # shape = N,D
            pred_score_array = (pred_score_array > 0).astype(np.int32)
            assert len(pred_score_array) == x.shape[0]
        return pred_score_array.astype(np.int32)  # return N x out_size, where N is number of nodes.

