from collections import defaultdict
import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F
import numpy as np
from graph_learning.model.st_attention_net.attention_base_block import MultiHeadAttention, \
    PositionwiseFeedForwardLayer, AttentionBlock


from graph_learning.model.open_crf.cython.factor_graph import FactorGraph
import random

from graph_learning.model.st_attention_net.enum_type import NoneModule, RecurrentType, NeighborMode, SpatialEdgeMode


class NodeRNN(chainer.Chain):

    def __init__(self, n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(NodeRNN, self).__init__()
        if not initialW:
            initialW = initializers.HeNormal()
        self.n_layer = n_layers
        if use_bi_lstm:
            assert outsize % 2 ==0
        with self.init_scope():
            if use_bi_lstm:
                self.lstm1 = L.NStepBiLSTM(self.n_layer, insize, 256,initialW=initialW, dropout=0.1) #dropout = 0.0
            else:
                self.lstm1 = L.NStepLSTM(self.n_layer, insize, 512, initialW=initialW,dropout=0.1)
            self.fc2 = L.Linear(512, 256, initialW=initialW)
            self.fc3 = L.Linear(256, 100, initialW=initialW)
            self.fc4 = L.Linear(100, outsize, initialW=initialW)

    def __call__(self, xs):
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hx = None
        cx = None
        # with chainer.no_backprop_mode():
        #     hx = chainer.Variable(xp.zeros((self.n_layer, len(xs), 512), dtype=xp.float32))
        #     cx = chainer.Variable(xp.zeros((self.n_layer, len(xs), 512), dtype=xp.float32))
        _, _, hs = self.lstm1(hx, cx, xs)  # hs is list of T x D variable
        # hs = [F.dropout(h) for h in hs]
        hs = [F.relu(self.fc2(h)) for h in hs]
        hs = [F.relu(self.fc3(h)) for h in hs]
        return [self.fc4(h) for h in hs]


class EdgeRNN(chainer.Chain):

    def __init__(self,n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(EdgeRNN, self).__init__()
        self.n_layer = n_layers
        self.outsize = outsize
        if use_bi_lstm:
            assert outsize % 2 == 0

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


class StAttentionNet(chainer.Chain):
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
                 neighbor_mode: NeighborMode = NeighborMode.concat_all, spatial_edge_model: SpatialEdgeMode = SpatialEdgeMode.all_edge,
                 recurrent_block_type: RecurrentType = RecurrentType.rnn, attn_heads=8, attn_dropout=0.1,
                 use_geometry_features=True, bi_lstm=False):
        super(StAttentionNet, self).__init__()
        self.spatial_edge_mode = spatial_edge_model
        self.use_geometry_features = use_geometry_features
        self.neighbor_mode = neighbor_mode
        self.out_size = out_size
        self.in_size = in_size
        self.frame_node_num = frame_node_num
        self.mid_size = 1024
        self.attn_dropout = attn_dropout
        self.batch = 1
        EdgeRecurrentModule = AttentionBlock if recurrent_block_type == RecurrentType.attention_block else EdgeRNN
        NodeRecurrentModule = AttentionBlock if recurrent_block_type == RecurrentType.attention_block else NodeRNN
        if recurrent_block_type == RecurrentType.no_temporal:
            EdgeRecurrentModule = PositionwiseFeedForwardLayer
            NodeRecurrentModule = PositionwiseFeedForwardLayer
        if spatial_edge_model == SpatialEdgeMode.no_edge:
            EdgeRecurrentModule = NoneModule

        with self.init_scope():
            if not initialW:
                initialW = initializers.HeNormal()

            self.bottom = dict()
                # TODO: G.factor_node can also meet the demand of all_pair connect situation
            if spatial_edge_model != SpatialEdgeMode.no_edge:
                for factor_node in G.factor_node:  # note that default all factor edge is all combinations of nodes
                    neighbors = factor_node.neighbor
                    print("constructing one edge RNN")
                    var_node_a = neighbors[0]
                    var_node_b = neighbors[1]
                    feature_len = 2 * self.mid_size
                    edge_module_id = ",".join(map(str, sorted([int(var_node_a.id), int(var_node_b.id)])))
                    if recurrent_block_type == RecurrentType.rnn:
                        self.add_link("Edge_{}".format(edge_module_id), EdgeRecurrentModule(n_layers,
                                                                                            feature_len,
                                                                                            self.mid_size, use_bi_lstm=bi_lstm))
                    else:
                        self.add_link("Edge_{}".format(edge_module_id), EdgeRecurrentModule(n_layers,
                                                                                        feature_len,
                                                                                        self.mid_size))
                    self.bottom[edge_module_id] = getattr(self,"Edge_{}".format(edge_module_id))  # 输出是node feature, 最后node feature concat起来
            else:
                for node_1 in G.var_node:
                    node_id_a = node_1.id
                    for node_2 in G.var_node:  # this bottom needs all contains self to self link
                        node_id_b = node_2.id
                        feature_len = 2 * self.mid_size
                        edge_module_id = ",".join(map(str, sorted([int(node_id_a), int(node_id_b)])))
                        self.add_link("Edge_{}".format(edge_module_id), EdgeRecurrentModule(n_layers,
                                                                                            feature_len,
                                                                                            self.mid_size))
                        self.bottom[edge_module_id] = getattr(self, "Edge_{}".format(edge_module_id))


            self.top = dict()
            self.mid_fc = L.Linear(self.in_size, self.mid_size, initialW=initialW)
            self.node_id_neighbor = defaultdict(list)

            # build top node RNN
            for idx, node in enumerate(G.var_node):
                print("constructing one node RNN")
                if spatial_edge_model == SpatialEdgeMode.all_edge:
                    assert len(node.neighbor) == len(G.var_node) - 1
                node_id = node.id
                assert node_id == idx
                # one is all weighted sum of all neighbors edge block output,
                # another is linear transformed node block output
                if neighbor_mode == NeighborMode.concat_all:
                    feature_len = self.mid_size * (len(node.neighbor)+1)
                elif neighbor_mode == NeighborMode.attention_fuse:
                    feature_len = self.mid_size
                elif neighbor_mode == NeighborMode.random_neighbor:
                    feature_len = self.mid_size * 2
                elif neighbor_mode == NeighborMode.no_neighbor:
                    assert spatial_edge_model == SpatialEdgeMode.no_edge
                    feature_len = self.mid_size

                # if spatial_edge_model == SpatialEdgeMode.configure_edge:
                #
                #     neighbors = node.neighbor
                #     for factor_node in neighbors:
                #         can_add = False
                #         for var_node in factor_node.neighbor:
                #             if var_node.id == node_id and not can_add:
                #                 can_add = True
                #                 continue
                #             self.node_id_neighbor[node_id].append(var_node.id)  # 这样应该也会将自己对自己相连的id包含在内
                #     for key, val_list in self.node_id_neighbor.items():
                #         self.node_id_neighbor[key] = sorted(
                #             val_list)  # id相当于行号，self.node_id_neighbor的目的是找出neighbor是谁，便于构建连接关系
                #
                if recurrent_block_type == RecurrentType.rnn:
                    self.add_link("Node_{}".format(node_id),
                                  NodeRecurrentModule(n_layers, feature_len, self.out_size, use_bi_lstm=bi_lstm))
                else:
                    self.add_link("Node_{}".format(node_id),
                                  NodeRecurrentModule(n_layers, feature_len, self.out_size))
                self.top[str(node_id)] = getattr(self, "Node_{}".format(node_id))

            self.node_id_neighbor = defaultdict(list)
            for node_id_a in self.top.keys():
                for node_id_b in self.top.keys():
                    node_id_a = int(node_id_a)
                    node_id_b = int(node_id_b)
                    joint_id = ",".join(map(str, sorted([node_id_a, node_id_b])))
                    if node_id_a != node_id_b and spatial_edge_model != SpatialEdgeMode.no_edge:
                        assert joint_id in self.bottom
                        self.node_id_neighbor[node_id_a].append(node_id_b)
                    elif spatial_edge_model == SpatialEdgeMode.no_edge:
                        self.node_id_neighbor[node_id_a].append(node_id_b)
            for node_id, node_neighbor_list in self.node_id_neighbor.items():
                self.node_id_neighbor[node_id] = sorted(node_neighbor_list)

            if neighbor_mode == NeighborMode.random_neighbor:
                self.change_neighbor_dict_randomly(self.node_id_neighbor)
            if neighbor_mode == NeighborMode.attention_fuse:
                # we want to use node feature to attend all neighbor feature
                self.head = attn_heads
                self.multi_head_attention_module = MultiHeadAttention(n_heads=attn_heads, d_model=self.mid_size,
                                                                    d_k=self.mid_size//attn_heads,
                                                                    d_v=self.mid_size//attn_heads)


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

    def no_fuse_geometry_feature_func(self, edge_feature, geo_encoded_feature, fetch_index):
        return edge_feature

    def fuse_geometry_feature_func(self, edge_feature, geo_encoded_feature, fetch_index):
        # geo_encoded_feature = (T, F^2, 2 * in_size)  edge_feature = (T, 2D) fetch_index = (T, 2)
        assert geo_encoded_feature.shape[1] == self.frame_node_num ** 2
        T = geo_encoded_feature.shape[0]
        first_frame_fetch_index = fetch_index[0] # shape = (2,)
        fetch_a, fetch_b = first_frame_fetch_index[0], first_frame_fetch_index[1]
        frame_offset = fetch_a * self.frame_node_num + fetch_b
        fetch_geo_index = np.array([frame_offset for _ in range(T)], dtype=np.int32)
        # shape = (T, mid_size * 2)  note that we must embedding to mid_size * 2 first
        geo_encoded_feature = geo_encoded_feature[np.arange(T).astype(np.int32), fetch_geo_index, :]
        return edge_feature + geo_encoded_feature


    def edge_recurrent_forward(self,xs, gs_encoded, cross_frame_node_dict, fuse_geometry_feature_func):
        edge_out_dict = dict()
        for edge_module_id, edge_module in self.bottom.items():
            orig_node_id_a, orig_node_id_b = edge_module_id.split(",")
            node_list_a = cross_frame_node_dict[int(orig_node_id_a)]
            node_list_b = cross_frame_node_dict[int(orig_node_id_b)]
            assert len(node_list_a) == len(node_list_b)  # len = T
            fetch_x_index = np.array(list(zip(node_list_a, node_list_b)))  # T x 2
            edge_feature = xs[fetch_x_index.flatten(), :]  # (batch x (T x 2)) x D
            edge_feature = edge_feature.reshape(fetch_x_index.shape[0],
                                                2 * xs.shape[-1])  # shape = (T, 2D)
            edge_feature = fuse_geometry_feature_func(edge_feature, gs_encoded, fetch_x_index)
            edge_out = edge_module([edge_feature])[0]  # shape = (T,D)
            node_id_a, node_id_b = list(map(int, edge_module_id.split(',')))
            edge_out_dict[edge_module_id] = {node_id_a: edge_out, node_id_b: edge_out}
        return edge_out_dict

    def node_recurrent_forward(self, xs, neighbor_out_dict, cross_frame_node_dict, fuse_feature_func):
        # xs shape = (N,D)
        node_out_list = []
        for node_module_id, node_module in sorted(self.top.items(), key=lambda e:int(e[0])):
            node_module_id = int(node_module_id)
            neighbor_node_id_list = self.node_id_neighbor[node_module_id]
            cross_frame_node_id_list = cross_frame_node_dict[node_module_id]  # length = T
            x = xs[cross_frame_node_id_list] # shape = (T,mid_size)
            x = F.expand_dims(x, axis=0)  # shape = (1, T, mid_size)
            concat_features = []
            for neighbor_node_id in neighbor_node_id_list:
                joint_id = ",".join(map(str, sorted([int(node_module_id), int(neighbor_node_id)])))
                neighbor_out_feature = neighbor_out_dict[joint_id][neighbor_node_id]  # shape = (T,mid_size)
                concat_features.append(neighbor_out_feature)
            concat_features = F.stack(concat_features)  # shape = (neighbor, T, mid_size)
            fuse_input = fuse_feature_func(x, concat_features)  # return shape = (T, D)
            node_out_list.append(node_module([fuse_input])[0])  # each shape = (T, out_size)
        node_output = F.stack(node_out_list)  # shape nodeRNN_num x T x out_size
        node_output = F.transpose(node_output, axes=(1, 0, 2))  # reorder, shape = T x nodeRNN_num x out_size
        node_output = node_output.reshape(-1, self.out_size)  # shape N x out_size
        return node_output

    def change_neighbor_dict_randomly(self, node_id_neighbor):

        for node_id, neighbor_node_id_list in node_id_neighbor.items():
            new_neighbor_node_id_list = [random.choice(neighbor_node_id_list)]
            node_id_neighbor[node_id] = new_neighbor_node_id_list


    def change_random_neighbor_out_dict(self, orig_neighbor_out_dict):
        for joint_id, out_dict in orig_neighbor_out_dict.items():
            for node_id, cross_time_feature in out_dict.items():
                new_cross_time_feature = []
                for time, time_slice_feature in enumerate(F.split_axis(cross_time_feature,
                                                       cross_time_feature.shape[0], axis=0, force_tuple=True)):
                    other_random_key = random.choice(list(orig_neighbor_out_dict.keys()))
                    other_inner_dict = orig_neighbor_out_dict[other_random_key]
                    other_inner_random_key = random.choice(list(other_inner_dict.keys()))
                    other_feature = other_inner_dict[other_inner_random_key] # T, D
                    new_cross_time_feature.append(other_feature[time])
                new_cross_time_feature = F.stack(new_cross_time_feature) # T,D
                out_dict[node_id] = new_cross_time_feature

    def node_neighbor_out_dict(self, xs, cross_frame_node_dict):
        # xs shape = (N,D)
        neighbor_out_dict = dict()
        for edge_module_id in self.bottom.keys():
            node_id_a, node_id_b = list(map(int, edge_module_id.split(',')))
            cross_frame_nodes_a = cross_frame_node_dict[node_id_a]  # length = T
            cross_frame_nodes_b = cross_frame_node_dict[node_id_b]  # length = T
            neighbor_out_dict[edge_module_id] = {node_id_a: xs[cross_frame_nodes_a], node_id_b: xs[cross_frame_nodes_b]}
        return neighbor_out_dict

    def neighbor_concat_fuse(self, x, z):
        # x shape = (1,T,D), z shape = (F',T,D), where F' is neighbor number of such specific x
        assert x.shape[2] == z.shape[2]
        assert x.shape[0] == 1
        out = F.concat((x, z), axis=0) # shape = (1+F'),T,D
        out = F.transpose(out, axes=(1, 0, 2)) # T, (1+F'), D
        return F.reshape(out, shape=(z.shape[1], (x.shape[0] + z.shape[0])*z.shape[2])) # shape = (T, (1+F')D)

    def no_neighbor_fuse(self, x, z):
        # x shape = (1,T,D)
        return F.reshape(x, shape=(x.shape[1], x.shape[2]))  # shape = (T,D)

    def neighbor_attention_fuse(self, x, z):
        # x shape = (1,T,D), z shape = (F',T,D) where F' is neighbor number of such x
        x = F.transpose(x,
                        axes=(1, 0, 2))  # shape = (T,1,D), where F can be seen as sequence length(node number in frame)
        z = F.transpose(z, axes=(1, 0, 2))  # shape = (T,F',D)
        attention_value = self.multi_head_attention_module(x, z,
                                                           z)  # shape = (T,1,D), note that dimension F is same order with x
        assert attention_value.shape == x.shape
        # fuse_input = F.concat((x, attention_value), axis=2) # shape = (T,1,2D)
        fuse_input = attention_value
        return F.squeeze(fuse_input) # shape = (T,D)

    def forward(self, xs, gs, crf_pact_structures):  # xs shape = (batch, N, D), batch forever = 1
        '''
        :param xs: appearance features of all boxes feature across all frames
        :param gs:  geometry features of all polygons. each is 4 coordinates represent box
        :param crf_pact_structures: packaged graph structure contains supplementary information
        :return:
        '''
        xp = chainer.cuda.get_array_module(xs.data)

        assert xs.shape[0] == self.batch
        xs = xs.reshape(-1, self.in_size) # shape = (batch * T * box_num, D)
        gs = gs.data.reshape(-1, 4)  # shape = (batch * T * box_num, 4)
        gs_encoded = None
        fuse_box_feature_func = self.no_fuse_geometry_feature_func
        if self.use_geometry_features:
            T = gs.shape[0] // self.frame_node_num
            gs = gs.reshape(T, self.frame_node_num, gs.shape[-1])
            g_1 = xp.tile(gs, (1, 1, self.frame_node_num))  # shape = (T, F, 4 * F)
            g_1 = xp.reshape(g_1,
                              (T, self.frame_node_num ** 2, gs.shape[-1]))  # after tile: (T, F, (4 x F)) then (T,F^2,4)
            g_2 = xp.tile(gs, (1, self.frame_node_num, 1))  # shape = (T, F*F, 4)
            gs_encoded = self.encode_box_offset(g_1, g_2)   # shape = (T, F*F, 4)
            gs_encoded = self.position_encoding(gs_encoded.reshape(-1, gs.shape[-1]), out_channel=2 * self.in_size)
            gs_encoded = xp.reshape(gs_encoded, (T, self.frame_node_num**2, 2*self.in_size))
            fuse_box_feature_func = self.fuse_geometry_feature_func



        T = xs.shape[0] // self.frame_node_num
        # first frame node_id ==> other frame node_id in same corresponding box
        cross_frame_node_dict = crf_pact_structures[0].nodeRNN_id_dict
        convert_dim_xs = F.leaky_relu(self.mid_fc(xs)) # shape = (batch * T * box_num, mid_size)
        if self.neighbor_mode == NeighborMode.concat_all:
            if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
                edge_out_dict = self.edge_recurrent_forward(convert_dim_xs, gs_encoded, cross_frame_node_dict, fuse_box_feature_func)
                node_output = self.node_recurrent_forward(convert_dim_xs, edge_out_dict, cross_frame_node_dict,
                                            self.neighbor_concat_fuse)
                edge_out_dict.clear()
                return F.expand_dims(node_output, axis=0)

            elif self.spatial_edge_mode == SpatialEdgeMode.no_edge:
                node_out_dict = self.node_neighbor_out_dict(convert_dim_xs, cross_frame_node_dict)
                node_output = self.node_recurrent_forward(convert_dim_xs, node_out_dict, cross_frame_node_dict,
                                            self.neighbor_concat_fuse)
                node_out_dict.clear()
                return F.expand_dims(node_output, axis=0)

        elif self.neighbor_mode == NeighborMode.attention_fuse:
            if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
                edge_out_dict = self.edge_recurrent_forward(convert_dim_xs, gs_encoded, cross_frame_node_dict, fuse_box_feature_func)
                node_output = self.node_recurrent_forward(convert_dim_xs, edge_out_dict, cross_frame_node_dict,
                                                          self.neighbor_attention_fuse)
                return F.expand_dims(node_output, axis=0)
            elif self.spatial_edge_mode == SpatialEdgeMode.no_edge:
                node_out_dict = self.node_neighbor_out_dict(convert_dim_xs, cross_frame_node_dict)
                node_output = self.node_recurrent_forward(convert_dim_xs, node_out_dict, cross_frame_node_dict,
                                                          self.neighbor_attention_fuse)
                node_out_dict.clear()
                return F.expand_dims(node_output, axis=0)

        elif self.neighbor_mode == NeighborMode.no_neighbor:
            assert self.spatial_edge_mode == SpatialEdgeMode.no_edge # this is special case; no_neighbor => no_edge, but no_edge cannot => no_neighbor
            if self.spatial_edge_mode == SpatialEdgeMode.no_edge:
                # shape = F, T, D
                node_out_dict = self.node_neighbor_out_dict(convert_dim_xs, cross_frame_node_dict)
                node_output = self.node_recurrent_forward(convert_dim_xs, node_out_dict, cross_frame_node_dict,
                                                          self.no_neighbor_fuse)
                node_out_dict.clear()
                return F.expand_dims(node_output, axis=0)

        elif self.neighbor_mode == NeighborMode.random_neighbor:
            if self.spatial_edge_mode == SpatialEdgeMode.all_edge:
                edge_out_dict = self.edge_recurrent_forward(convert_dim_xs, gs_encoded, cross_frame_node_dict, fuse_box_feature_func)
                self.change_random_neighbor_out_dict(edge_out_dict)

                node_output = self.node_recurrent_forward(convert_dim_xs, edge_out_dict, cross_frame_node_dict,
                                                            self.neighbor_concat_fuse)
                edge_out_dict.clear()
                return F.expand_dims(node_output, axis=0)  # shape = B,N,out_size

            elif self.spatial_edge_mode == SpatialEdgeMode.no_edge:
                node_out_dict = self.node_neighbor_out_dict(convert_dim_xs, cross_frame_node_dict)
                self.change_random_neighbor_out_dict(node_out_dict)

                node_output = self.node_recurrent_forward(convert_dim_xs, node_out_dict, cross_frame_node_dict,
                                                          self.neighbor_concat_fuse)
                node_out_dict.clear()
                return F.expand_dims(node_output, axis=0) # shape = B,N,out_size



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

