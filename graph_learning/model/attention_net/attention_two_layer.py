import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F
from itertools import combinations
from graph_learning.model.attention_net.attention_base_block import AttentionBlock
from graph_learning.model.open_crf.cython.factor_graph import FactorGraph
import numpy as np


class AttentionTwoLayer(chainer.Chain):
    def __init__(self, G:FactorGraph, n_layers:int, in_size:int, out_size:int, frame_node_num:int, initialW=None,
                 neighbor_mode="concat", attn_heads=1, attn_dropout=0.5):
        super(AttentionTwoLayer, self).__init__()
        assert neighbor_mode in ['concat', 'weight_sum']
        self.neighbor_mode = neighbor_mode
        self.out_size = out_size
        self.in_size = in_size
        self.frame_node_num = frame_node_num
        self.mid_size = 512
        self.attn_dropout = attn_dropout
        with self.init_scope():
            if not initialW:
                initialW = initializers.HeNormal()
            self.bottom = dict()
            self.top = dict()
            self.mid_fc = L.Linear(self.in_size, self.mid_size, initialW=initialW)
            for node_a, node_b in combinations(G.var_node, 2):  # iterater over all combinations of nodes pair
                feature_len = 2 * in_size
                node_id_a = node_a.id
                node_id_b = node_b.id
                edge_module_id = ",".join(map(str, sorted([int(node_id_a), int(node_id_b)])))
                self.add_link("Edge_{}".format(edge_module_id), AttentionBlock(n_layers, feature_len, self.mid_size))
                self.bottom[edge_module_id] = getattr(self, "Edge_{}".format(edge_module_id))

            # build top node RNN
            for node in G.var_node:
                assert len(node.neighbor) == len(G.var_node) - 1
                node_id = node.id
                # one is all weighted sum of all neighbors edge block output,
                # another is linear transformed node block output
                if neighbor_mode == 'concat':
                    feature_len = self.mid_size * (len(node.neighbor)+1)
                elif neighbor_mode == 'weight_sum':
                    feature_len = self.mid_size * 2
                self.add_link("Node_{}".format(node_id),
                              AttentionBlock(n_layers, feature_len, self.out_size))
                self.top[str(node_id)] = getattr(self, "Node_{}".format(node_id))
            if neighbor_mode == "weight_sum":
                self.attn_kernels = []
                self.convert_dim_kernels = []
                for head in range(attn_heads):
                    attn_fc = L.Linear(self.mid_size, 1)
                    self.add_link("attn_fc_{}".format(head), attn_fc)
                    self.attn_kernels.append("attn_fc_{}".format(head))
                    convert_dim_fc = L.Linear(self.in_size, self.mid_size)
                    self.add_link("dim_fc_{}".format(head), convert_dim_fc)
                    self.convert_dim_kernels.append("dim_fc_{}".format(head))

    def output(self, xs, crf_pact_structures):  # xs shape = (batch, N, D), batch forever = 1
        xp = chainer.cuda.get_array_module(xs.data)
        batch = 1
        assert xs.shape[0] == batch
        xs = xs.reshape(-1, self.in_size) # shape = (batch * T * box_num, D)
        cross_frame_node_dict = crf_pact_structures[0].nodeRNN_id_dict  # first frame node_id ==> other frame node_id in same box
        edge_out_dict = dict()
        for edge_module_id, edge_module in self.bottom.items():
            orig_node_id_a, orig_node_id_b = edge_module_id.split(",")
            node_list_a = cross_frame_node_dict[int(orig_node_id_a)]
            node_list_b = cross_frame_node_dict[int(orig_node_id_b)]
            assert len(node_list_a) == len(node_list_b)  # len = T
            fetch_x_index = np.array(list(zip(node_list_a, node_list_b)))  # T x 2
            edge_feature = xs[fetch_x_index.flatten(), :]  # (batch x (T x 2)) x D
            edge_feature = edge_feature.reshape(batch, fetch_x_index.shape[0], 2 * xs.shape[-1])  # shape = (batch, T, 2D)
            edge_feature = F.transpose(edge_feature, (0,2,1)) # shape = (batch, 2D, T)
            edge_out = edge_module(edge_feature, None)
            edge_out = F.reshape(edge_out, (self.mid_size, -1))
            edge_out_dict[edge_module_id] = edge_out # return shape = D, T

        node_output = []
        if self.neighbor_mode == "concat":

            for node_module_id, node_module in sorted(self.top.items(), key=lambda e: int(e[0])):
                concat_features = []
                node_list = cross_frame_node_dict[int(node_module_id)]  # length = T
                converted_node_feature = self.mid_fc(xs[node_list, :]) # xs = (T, mid_size)
                converted_node_feature = F.transpose(converted_node_feature) # shape = (D, T) where D = mid_size
                concat_features.append(converted_node_feature)
                # all nodes will be neighbors except itself
                for node_module_neighbor_id, node_module_neighbor in sorted(self.top.items(), key=lambda e: int(e[0])):
                    if node_module_id == node_module_neighbor_id:
                        continue
                    edge_combine_id = ",".join(map(str, sorted([int(node_module_id), int(node_module_neighbor_id)])))
                    edge_out = edge_out_dict[edge_combine_id]
                    concat_features.append(edge_out)
                frame_num = len(node_list) # T
                concat_features = F.stack(concat_features, axis=0)  # shape = (neighbor + 1) x D x T
                assert concat_features.shape[1] == self.mid_size
                concat_features = F.reshape(concat_features, (len(self.top) * self.mid_size, frame_num))  # shape = ((neighbor + 1) x D) x T
                concat_features = F.expand_dims(concat_features, 0) # shape = 1 x ((neighbor + 1) x D) x T
                node_output.append(node_module(concat_features, None)) #  return shape = 1, D, T
            edge_out_dict.clear()  # save GPU memory

        elif self.neighbor_mode == 'weight_sum':
            # construct N x N x F' matrix, where F' comes from neighbor edge_module output
            all_neighbor_matrix = []
            frame_num = 0

            for node_module_id, node_module in sorted(self.top.items(), key=lambda e: int(e[0])):
                node_list = cross_frame_node_dict[int(node_module_id)]  # length = T
                if frame_num != 0:
                    assert len(node_list) == frame_num, "frame_num = {0} != {1}".format(frame_num, len(node_list))
                frame_num = len(node_list)
                for node_module_neighbor_id, node_module_neighbor in sorted(self.top.items(), key=lambda e: int(e[0])):
                    if node_module_id == node_module_neighbor_id:
                        # the diagonal value will be erase later
                        all_neighbor_matrix.append(chainer.Variable(self.xp.full((self.mid_size, frame_num), -np.inf)))
                        continue
                    edge_combine_id = ",".join(map(str, sorted([int(node_module_id), int(node_module_neighbor_id)])))
                    edge_out = edge_out_dict[edge_combine_id]
                    all_neighbor_matrix.append(edge_out)  # each element is (D, T)
            edge_out_dict.clear()
            all_neighbor_matrix = F.stack(all_neighbor_matrix)  # shape = N^2 x D x T
            all_neighbor_matrix = F.transpose(all_neighbor_matrix, (2,0,1)) # shape = T x N^2 x D
            assert frame_num == all_neighbor_matrix.shape[0], "frame :{0} != {1}".format(frame_num,
                                                                                        all_neighbor_matrix.shape[0])
            assert len(self.top) * len(self.top) == all_neighbor_matrix.shape[1], "all node module count"
            assert self.mid_size == all_neighbor_matrix.shape[2]
            # construct over
            all_neighbor_matrix = F.reshape(all_neighbor_matrix, (all_neighbor_matrix.shape[0]*all_neighbor_matrix.shape[1],
                                                                  self.mid_size)) # shape = (T x N^2) x D

            all_neighbor_output = []
            mid_transfer_x = self.mid_fc(xs)
            mid_transfer_x = F.reshape(mid_transfer_x, (frame_num, len(self.top), self.mid_size))  # shape = (T, N, mid_size)
            for idx, attn_layer_name in enumerate(self.attn_kernels):
                convert_dim_fc_name = self.convert_dim_kernels[idx]
                convert_dim_fc = getattr(self, convert_dim_fc_name)
                linear_transfer_x = convert_dim_fc(xs)  # all_N x mid_size
                linear_transfer_x = F.reshape(linear_transfer_x,
                                              (frame_num, len(self.top), self.mid_size))  # shape = T x N x mid_size

                dense = getattr(self, attn_layer_name)(all_neighbor_matrix)
                dense = F.squeeze(dense) # one dimension vetor,  shape = (T x N^2),
                # reshape to T x N x N
                dense = F.reshape(dense, shape=(frame_num, len(self.top), len(self.top)))
                softmax_val = F.softmax(F.leaky_relu(dense), axis=2)
                # push nan to zero
                softmax_val = F.where(self.xp.isnan(softmax_val), self.xp.zeros(softmax_val.shape, dtype="f"), softmax_val)
                dropout_val = F.dropout(softmax_val, ratio=self.attn_dropout)  # shape =(T, N, N)
                # weight sum of all neighbor edge output value, 可见不是将edge_module的输出信息直接concat用作node_module的输入信息，而是作为了权重一般的存在，这个权重，最终与x去乘
                neighbor_features = F.matmul(dropout_val, linear_transfer_x)  # batch(T) matmul of (N x N) x (N x mid_size) = T x N x mid_size
                neighbor_features = F.leaky_relu(neighbor_features)  # shape = (T, N, mid_size)
                all_neighbor_output.append(neighbor_features)
            all_neighbor_output = F.mean(F.stack(all_neighbor_output), axis=0)  # shape = (T, N, mid_size)

            # concat neighbor(weighted sum) and self convert dimension x ,then input to node_module
            node_input = F.concat((mid_transfer_x, all_neighbor_output), axis=2)  # shape= (T, N, 2 * mid_size)
            node_input = F.reshape(node_input, (node_input.shape[0] * node_input.shape[1], 2 * self.mid_size)) # shape = (all_N, 2 * mid_size)
            for node_module_id, node_module in sorted(self.top.items(), key=lambda e: int(e[0])):
                node_list = cross_frame_node_dict[int(node_module_id)]  # length = T
                node_module_input = node_input[node_list, :]  # shape = (T, 2 * mid_size)
                node_module_input = F.expand_dims(F.transpose(node_module_input), 0) # shape = (1, 2 * mid_size, T)
                node_output.append(node_module(node_module_input, None))

        node_output = F.stack(node_output)  # shape = node_module_num x 1 x out_size x T
        node_output = F.squeeze(node_output, 1)  # shape = node_module_num x out_size x T
        node_output = F.transpose(node_output, axes=(2, 0, 1))  # reorder to T x node_module_num x out_size, 这样意味着node_RNN的排序要与文件中一致
        node_output = node_output.reshape(1, -1, self.out_size)  # shape = batch x N x out_size
        return node_output
