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

class NodeRNN(chainer.Chain):

    def __init__(self, n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(NodeRNN, self).__init__()
        if not initialW:
            initialW = initializers.HeNormal()
        self.n_layer = n_layers

        with self.init_scope():
            if use_bi_lstm:
                self.lstm = L.NStepBiLSTM(self.n_layer, 1024, outsize//2,initialW=initialW, dropout=0.1) #dropout = 0.0
            else:
                self.lstm = L.NStepLSTM(self.n_layer, 1024, outsize, initialW=initialW,dropout=0.1)
            self.fc1 = L.Linear(insize, 1024, initialW=initialW)
            self.fc2 = L.Linear(1024, 1024, initialW=initialW)

    def __call__(self, xs):  # input list of T,D
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hx = None
        cx = None

        hs = [F.relu(self.fc1(h)) for h in xs]
        hs = [F.relu(self.fc2(h)) for h in hs]
        _, _, hs = self.lstm(hx, cx, hs)  # hs is list of T x D variable
        return hs


class ConnLabelRNN(chainer.Chain):
    def __init__(self, n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(ConnLabelRNN, self).__init__()
        if not initialW:
            initialW = initializers.HeNormal()
        self.n_layer = n_layers
        with self.init_scope():
            if use_bi_lstm:
                self.lstm1 = L.NStepBiLSTM(self.n_layer, insize, 512, initialW=initialW, dropout=0.1)
            else:
                self.lstm1 = L.NStepLSTM(self.n_layer, insize, 1024, initialW=initialW, dropout=0.1)
            self.fc2 = L.Linear(1024, 512)
            self.fc3 = L.Linear(512, outsize)

    def __call__(self, xs):
        _,_,hs = self.lstm1(None, None, xs)
        hs = [F.relu(self.fc2(h)) for h in hs]
        hs = [self.fc3(h) for h in hs]
        return hs

class EdgeRNN(chainer.Chain):

    def __init__(self,n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(EdgeRNN, self).__init__()
        self.n_layer = n_layers
        self.outsize = outsize
        if use_bi_lstm:
            assert outsize % 2 == 0, outsize

        if not initialW:
            initialW = initializers.HeNormal()

        with self.init_scope():
            self.fc1 = L.Linear(insize, 256, initialW=initialW)
            self.fc2 = L.Linear(256, 256, initialW=initialW)
            if use_bi_lstm:
                self.lstm3 = L.NStepBiLSTM(self.n_layer, 256, outsize//2, initialW=initialW, dropout=0.1)  #dropout = 0.0
            else:
                self.lstm3 = L.NStepLSTM(self.n_layer, 256, outsize,initialW=initialW, dropout=0.1)

    def __call__(self, xs):
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hs = [F.relu(self.fc1(x)) for x in xs]
        hs = [F.relu(self.fc2(h)) for h in hs]
        hx = None
        cx = None
        # hx = chainer.Variable(xp.zeros((self.n_layer, len(xs), self.outsize), dtype=xp.float32))
        # cx = chainer.Variable(xp.zeros((self.n_layer, len(xs), self.outsize), dtype=xp.float32))

        _, _, hs = self.lstm3(hx, cx, hs)
        # https://docs.chainer.org/en/stable/reference/core/configuration.html?highlight=config and https://stackoverflow.com/questions/45757330/how-to-use-chainer-using-config-to-stop-f-dropout-in-evaluate-predict-process-in
        # hs = [F.dropout(h) for h in hs]
        return hs


class ReverseSrnnNet(chainer.Chain):
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
        super(ReverseSrnnNet, self).__init__()
        self.neg_pos_ratio = 3
        self.database = database
        self.spatial_edge_mode = spatial_edge_model
        self.out_size = out_size
        self.in_size = in_size
        self.frame_node_num = config.BOX_NUM[self.database]
        self.mid_size = 1024
        ConnLabelRecurrentModule = AttentionBlock if recurrent_block_type == RecurrentType.attention_block else ConnLabelRNN
        NodeRecurrentModule = AttentionBlock if recurrent_block_type == RecurrentType.attention_block else NodeRNN
        if recurrent_block_type == RecurrentType.no_temporal:
            ConnLabelRecurrentModule = PositionwiseFeedForwardLayer
            NodeRecurrentModule = PositionwiseFeedForwardLayer

        with self.init_scope():
            if not initialW:
                initialW = initializers.HeNormal()

            self.bottom = dict()
            for i in range(self.frame_node_num):
                if recurrent_block_type == RecurrentType.rnn:
                    self.add_link("Node_{}".format(i),
                                  NodeRNN(n_layers, self.in_size, self.mid_size, use_bi_lstm=bi_lstm))
                else:
                    self.add_link("Node_{}".format(i),
                                  NodeRecurrentModule(n_layers, self.in_size, self.mid_size))
                self.bottom[str(i)] = getattr(self, "Node_{}".format(i))
            self.node_classify_fc = L.Linear(self.mid_size, self.out_size, initialW=initialW)
            self.conn_transform_dim_fc = L.Linear(2 * self.mid_size, self.mid_size, initialW=initialW)
            self.top = dict()
            if spatial_edge_model != SpatialEdgeMode.no_edge:
                for node_a_str, node_b_str in combinations(list(self.bottom.keys()), 2):  # note that edge is all combinations of nodes
                    joint_id = ",".join(map(str, sorted([int(node_a_str), int(node_b_str)])))
                    if recurrent_block_type == RecurrentType.rnn:
                        self.add_link("Conn_{}".format(joint_id), ConnLabelRNN(n_layers,self.mid_size,
                                                                                            self.out_size, use_bi_lstm=bi_lstm))
                    else:
                        self.add_link("Conn_{}".format(joint_id), ConnLabelRecurrentModule(n_layers,
                                                                                        self.mid_size,
                                                                                        self.out_size))
                    self.top[joint_id] = getattr(self,"Conn_{}".format(joint_id))


    def encode_box_offset(self, box1, box2):
        xp = self.xp
        if box1.ndim == 2 and box2.ndim == 2:  # shape = N, 4
            box1_x1y1_x2y2 = xp.reshape(box1, (-1, 2, 2))
            box1_x1y1, box1_x2y2 = xp.split(box1_x1y1_x2y2, 2, axis=1) # shape = (N,1,2)
            box1_w_h = box1_x2y2 - box1_x1y1  # shape = (N,1,2)
            box1_x_y = (box1_x2y2 + box1_x1y1) * 0.5 # shape = (N,1,2)

            box2_x1y1_x2y2 = xp.reshape(box2, (-1,2,2))
            box2_x1y1, box2_x2y2= xp.split(box2_x1y1_x2y2, 2, axis=1)  # shape = (N,1,2)
            box2_w_h = box2_x2y2 - box2_x1y1  # shape = (N,1,2)
            box2_x_y = (box2_x2y2 + box2_x1y1) * 0.5  # shape = (N,1,2)


            txty = xp.log(xp.abs(box2_x_y - box1_x_y) / box1_w_h )  # shape = (N,1,2)
            twth = xp.log(box2_w_h/ box1_w_h ) # shape = (N,1,2)
            encoded = xp.concatenate([txty, twth], axis=1) # (N,2,2)
            return encoded.reshape(-1, 4)  # the same as paper formula

        elif box1.ndim == 3 and box2.ndim == 3:  # shape = T, F^2, 4
            T = box1.shape[0]
            assert T == box2.shape[0]
            assert self.frame_node_num**2 == box1.shape[1]
            box1_x1y1_x2y2 = xp.reshape(box1, (T, box1.shape[1], 2, 2)) # T,F^2,2,2
            box1_x1y1, box1_x2y2 = xp.split(box1_x1y1_x2y2, 2, axis=2) # T, F^2, 1, 2
            box1_w_h = box1_x2y2 - box1_x1y1  # shape = (T, F^2, 1, 2)
            box1_x_y = (box1_x2y2 + box1_x1y1) * 0.5 # shape = (T, F^2, 1, 2)

            box2_x1y1_x2y2 = xp.reshape(box2, (T, box2.shape[1], 2, 2)) # T, F^2, 2, 2
            box2_x1y1, box2_x2y2 = xp.split(box2_x1y1_x2y2, 2, axis=2)  # T, F^2, 1, 2
            box2_w_h = box2_x2y2 - box2_x1y1  # shape = (T, F^2, 1, 2)
            box2_x_y = (box2_x2y2 + box2_x1y1) * 0.5 # shape = (T, F^2, 1, 2)

            txty = xp.log(xp.abs(box2_x_y - box1_x_y) / box1_w_h)  # shape = (T, F^2, 1, 2)
            twth = xp.log(box2_w_h / box1_w_h)  # shape = (T, F^2, 1, 2)
            encoded = xp.concatenate([txty, twth], axis=2)  # (T, F^2 ,2,2)
            return encoded.reshape(T, self.frame_node_num**2, 4)


    def position_encoding(self, low_dim_data, out_channel):
        # e.g. low_dim_data = N x 4 where 4 is coordinates number of box
        pieces = low_dim_data.shape[1]
        assert (out_channel % (2 * pieces) == 0)
        xp = self.xp
        num_timescales = out_channel // (2 * pieces)
        log_timescale_increment = (
                xp.log(10000. / 1.) / (float(num_timescales) - 1))  # float(num_timescales) - 1 = paper's d_model
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * -log_timescale_increment)  # shape= (num_timescales,)
        signal = []
        for piece in range(pieces):
            scaled_time = \
                xp.expand_dims(low_dim_data[:, piece], 1) * xp.expand_dims(inv_timescales,
                                                                           0)  # shape = (N, num_timescales)
            signal_piece = xp.concatenate(
                [xp.sin(scaled_time), xp.cos(scaled_time)],
                axis=1)  # shape = (N, 2 * num_timescales) = (N, out_channel//pieces)
            signal.append(signal_piece)
        signal = xp.concatenate(signal, axis=1)  # shape = (N, out_channel) = (N, d_g)
        return signal


    def node_recurrent_forward(self,xs): # xs is shape of (batch, T,F,D)
        node_out_dict = OrderedDict()
        for node_module_id, node_module in self.bottom.items():
            input_x = xs[:, :, int(node_module_id), :]  # B, T, D
            input_x = F.split_axis(input_x, input_x.shape[0], axis=0, force_tuple=True)
            input_x = [F.squeeze(x) for x in input_x]  # list of T,D
            node_out_dict[node_module_id] = F.stack(node_module(input_x)) # B, T, mid_size
        return node_out_dict

    def conn_recurrent_forward(self, node_out_dict):
        # xs shape = (N,D)
        conn_out_dict = dict()
        for conn_module_id, conn_module in self.top.items():
            node_module_id_a, node_module_id_b = conn_module_id.split(",")
            node_a_out = node_out_dict[node_module_id_a]  # B, T, D
            node_b_out = node_out_dict[node_module_id_b]  # B, T, D
            input = F.concat((node_a_out, node_b_out), axis=2) # B, T, 2D
            batch_size, seq_len, dim = input.shape
            input = F.reshape(input, (-1, dim))
            input = self.conn_transform_dim_fc(input)
            input = F.reshape(input, (batch_size, seq_len, self.mid_size))
            input = list(F.separate(input, axis=0))  # list of T, D
            conn_out_dict[conn_module_id] = F.stack(conn_module(input)) # B, T, D
        return conn_out_dict


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
        # first frame node_id ==> other frame node_id in same corresponding box
        node_out_dict = self.node_recurrent_forward(xs)
        # shape = F, B, T, mid_size
        node_out = F.stack([node_out_ for _, node_out_ in sorted(node_out_dict.items(),
                                                                 key=lambda e: int(e[0]))])

        node_out = F.transpose(node_out, (1,2,0,3))# shape = (B,T,F,D)
        assert self.frame_node_num == node_out.shape[2],node_out.shape[2]
        assert self.mid_size == node_out.shape[-1]
        assert T == node_out.shape[1]
        node_out = F.reshape(node_out, shape=(-1, self.mid_size))
        node_out = self.node_classify_fc(F.relu(node_out))  # shape = (N, out_size)
        node_out = F.reshape(node_out, shape=(batch, T, self.frame_node_num, self.out_size))
        if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
            conn_out_dict = self.conn_recurrent_forward(node_out_dict)
            return node_out, conn_out_dict
        return node_out


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
        conn_pred = []
        ts = []
        loss = 0
        accuracy = 0
        if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
            node_out, conn_out_dict = self.forward(xs)  # node_out B,T,F,D ,conn_out_dict each entry: B,T,D
            node_out = F.reshape(node_out, (-1, self.out_size))
            node_labels = self.xp.reshape(labels, (-1, self.out_size))
            pick_index, accuracy_pick_index = self.get_loss_index(node_out, node_labels)
            loss += F.sigmoid_cross_entropy(node_out[list(pick_index[0]), list(pick_index[1])],
                                            node_labels[list(pick_index[0]), list(pick_index[1])])
            accuracy_node = F.binary_accuracy(node_out[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                              node_labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])
            for conn_id, conn_out in conn_out_dict.items(): #
                node_id_a, node_id_b = tuple(map(int, conn_id.split(",")))
                label_a = labels[:, :, node_id_a, :]  # B,T,class
                label_b = labels[:, :, node_id_b, :]  # B,T,class
                # conn_label = self.xp.bitwise_or(label_a, label_b)  # B, T, class
                conn_label = label_a|label_b
                conn_pred.extend(conn_out)  # output will be list of (T, num_class) Variable
                ts.extend(conn_label)  # will be list of (T, num_class)
            conn_pred = F.stack(conn_pred) # connect_num * batch,T, class_num
            ts = self.xp.stack(ts) # connect_num * batch,T, class_num
            conn_pred = conn_pred.reshape(-1, self.out_size)
            ts = ts.reshape(-1, self.out_size)
            pick_index, accuracy_pick_index = self.get_loss_index(conn_pred, ts)
            accuracy_conn = F.binary_accuracy(conn_pred[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         ts[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])
            loss += F.sigmoid_cross_entropy(conn_pred[list(pick_index[0]), list(pick_index[1])],
                                           ts[list(pick_index[0]), list(pick_index[1])])
        elif self.spatial_edge_mode == SpatialEdgeMode.no_edge:
            node_out = self.forward(xs)
            node_out = F.reshape(node_out, (-1, self.out_size))
            node_labels = self.xp.reshape(labels, (-1, self.out_size))
            loss += F.sigmoid_cross_entropy(node_out, node_labels)
            pick_index, accuracy_pick_index = self.get_loss_index(node_out, node_labels)
            accuracy = F.binary_accuracy(node_out[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         node_labels[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])

        report_dict = {'loss': loss, 'accuracy_node': accuracy_node, "accuracy_conn":accuracy_conn}
        chainer.reporter.report(report_dict,
                                self)
        return loss

    # can only predict one frame based on previous T-1 frame feature
    def predict(self, xs): # all shape is (B, T, F, D)
        if not isinstance(xs, chainer.Variable):
            xs = chainer.Variable(xs)
        xp = chainer.cuda.cupy.get_array_module(xs)
        with chainer.no_backprop_mode():
            if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
                node_out, conn_out_dict = self.forward(xs) # node_out is B,T,F,class_num
                node_out = chainer.cuda.to_cpu(node_out.data) # B, T, F, class_num
                node_out = np.bitwise_or.reduce(node_out, axis=2)  # B, T, class_num
                temp_conn_output = []
                for conn_out in conn_out_dict.values():  # each is B,T,D
                    temp_conn_output.append(conn_out) # F, B, T, D
                temp_conn_output =F.transpose(F.stack(temp_conn_output), (1,2,0,3))  # B, T, conn_F, D
                temp_conn_output = chainer.cuda.to_cpu(temp_conn_output.data) # B,T, conn_F,D
                temp_conn_output = np.bitwise_or.reduce(temp_conn_output, axis=2)  #  B, T, D

                pred = node_out | temp_conn_output  # B, class_num
                pred = (pred > 0).astype(np.int32)
            else:
                node_out = self.forward(xs)  # node_out is B,T,F,class_num
                node_out = chainer.cuda.to_cpu(node_out.data)  # B, T, F, class_num
                node_out = np.bitwise_or.reduce(node_out, axis=2)  # B,T, class_num
                pred = (node_out > 0).astype(np.int32)
        return pred  # return batch x out_size, it is last time_step frame of 2-nd axis of input xs prediction

