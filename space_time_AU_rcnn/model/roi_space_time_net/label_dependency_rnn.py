from collections import defaultdict


import config
import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np

class BiDirectionLabelDependencyLSTM(chainer.Chain):
    def __init__(self, insize, outsize, class_num, label_win_size=3, x_win_size=1, train_mode=True, is_pad=True,
                 dropout_ratio=0.4):
        super(BiDirectionLabelDependencyLSTM, self).__init__()
        mid_size = 512
        self.out_size = outsize
        self.mid_size = mid_size
        with self.init_scope():
            self.forward_ld_rnn = LabelDependencyLSTM(insize, mid_size, class_num, label_win_size,
                                                      x_win_size, train_mode, is_pad, dropout_ratio=dropout_ratio)
            self.backward_ld_rnn = LabelDependencyLSTM(insize, mid_size, class_num, label_win_size,
                                                       x_win_size, train_mode, is_pad,dropout_ratio=dropout_ratio)
            self.fusion_fc = L.Linear(mid_size * 2, outsize)

    def __call__(self, xs, labels):
        xs = F.stack(xs)  # B, T, D
        labels = F.stack(labels)
        forward_out = self.forward_ld_rnn(xs, labels)

        reverse_xs = F.flip(xs, axis=1)
        reverse_label = F.flip(labels, axis=1)
        backward_out = self.backward_ld_rnn(reverse_xs, reverse_label)
        forward_out = F.stack(forward_out)  # B, T, D
        backward_out = F.stack(backward_out) # B, reverse_T, D
        backward_out = F.flip(backward_out, axis=1)
        bi_out = F.concat((forward_out, backward_out), axis=2)
        bi_out = F.reshape(bi_out, (xs.shape[0] * xs.shape[1], 2 * self.mid_size))
        final_out = self.fusion_fc(bi_out)
        final_out = F.reshape(final_out, (xs.shape[0], xs.shape[1], self.out_size))
        return list(F.separate(final_out, axis=0))


class LabelDependencyLSTM(chainer.Chain):
    # if we set pad=False, the returned value of axis T is smaller than input Variable
    def __init__(self, insize, outsize, class_num, label_win_size=3, x_win_size=1, train_mode=True, is_pad=True, dropout_ratio=0.4):
        super(LabelDependencyLSTM, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.train_mode = train_mode
        self.label_win_size = label_win_size
        self.x_win_size = x_win_size
        assert x_win_size % 2 == 1
        self.pad = is_pad
        self.mid_size = 512
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.label_embed = L.EmbedID(class_num+1, insize, ignore_label=-1, initialW=I.Uniform(1. / insize))
            self.mid_fc = L.Linear((label_win_size + x_win_size) * insize, self.mid_size)
            self.lstm = L.LSTM(self.mid_size, outsize)

    def clear(self):
        self.lstm.reset_state()

    def make_label_embedding(self, batch_labels):
        # labels shape = (batch, N, class_number)
        # out shape = (batch, N, embed_length)
        xp = chainer.cuda.cupy.get_array_module(batch_labels.data)
        batch_size, N, class_number = batch_labels.shape
        batch_embeded = []
        lengths = []
        label_dict = defaultdict(dict)  # all_label_id position -> labels index
        batch_label_id_list = []
        for i in range(batch_size):
            labels = batch_labels[i]
            each_label_id_list = []
            batch_length = 0
            for label_index, label in enumerate(labels.data):
                non_indexes = xp.nonzero(label)[0]
                if len(non_indexes) > 0:
                    for idx in range(len(non_indexes)):
                        non_idx = non_indexes[idx]
                        label_dict[i][len(each_label_id_list)] = label_index
                        each_label_id_list.append(non_idx + 1)
                        batch_length += 1
                else:
                    label_dict[i][len(each_label_id_list)] = label_index
                    each_label_id_list.append(0)
                    batch_length += 1
            lengths.append(batch_length)
            batch_label_id_list.extend(each_label_id_list)
        batch_label_id_list = xp.array(batch_label_id_list, dtype="i")
        batch_label_embeded = self.label_embed(batch_label_id_list)
        batch_label_embeded = F.split_axis(batch_label_embeded, np.cumsum(lengths)[:-1], axis=0)
        assert len(batch_label_embeded) == batch_size
        for batch_index, label_embeded in enumerate(batch_label_embeded):
            second_embeded_dict = defaultdict(list)
            for idx, embed_vector in enumerate(F.separate(label_embeded, axis=0)):
                label_index = label_dict[batch_index][idx]
                second_embeded_dict[label_index].append(embed_vector)
            final_embeded = []
            for batch_index, embed_vector_list in sorted(second_embeded_dict.items(), key=lambda e:int(e[0])):
                embed_vector_list = F.stack(embed_vector_list)
                final_embeded.append(F.sum(embed_vector_list, axis=0))
            final_embeded = F.stack(final_embeded)
            batch_embeded.append(final_embeded)
        return F.stack(batch_embeded)  # batch, N, embeded_length

    # input list of T,D where T includes padded win_width, return list of T-win_size, D
    # 先不考虑前面输入 其他clip的衔接帧的设计，就pad 全0
    def __call__(self, xs, labels):  # xs is list of T,D, labels is list of (T,12)
        self.clear()
        xs = F.stack(xs)  # B, T, D
        T = xs.shape[1]
        labels = F.stack(labels)  # B,T,D
        assert T == labels.shape[1]
        xp = chainer.cuda.get_array_module(xs.data)
        if self.pad:
            xs = F.pad(xs, pad_width=((0,0), (self.label_win_size, self.x_win_size//2), (0,0)),
                       mode='constant', constant_values=0.0)
            orig_label_type = labels.data.dtype
            labels.data = labels.data.astype('f')
            labels = F.pad(labels, pad_width=((0,0), (self.label_win_size, self.x_win_size//2), (0,0)),
                           mode='constant', constant_values=0.0)
            labels.data.astype(orig_label_type)

        xs = F.transpose(xs, (1, 0, 2))  # T, B, D
        batch_size, _, _ = labels.shape

        if self.train_mode:
            embeded_matrix = self.make_label_embedding(labels)  # Batch, T, embeded_length
            embeded_matrix = F.dropout(embeded_matrix, ratio=self.dropout_ratio)
            embeded_matrix_transpose = F.transpose(embeded_matrix, (1,0,2)) # T, batch_size, embeded_length
            # to use win_size previous gt_labels
            all_output = []
            for i in range(self.label_win_size, embeded_matrix_transpose.shape[0]-self.x_win_size//2): # will produce T step, each look previous win_size embeded
                input_label_embeded = embeded_matrix_transpose[i-self.label_win_size: i]  # shape = win_size, batch_size, embeded_length
                offset = np.concatenate([np.arange(-(self.x_win_size//2), 0), np.arange(0, self.x_win_size//2 + 1)])
                window_index = offset + i
                window_index = np.maximum(window_index, 0)
                assert window_index.shape[0] == self.x_win_size
                input_x_feature = xs[window_index]  # shape = win_size, B, D
                input_label_embeded = F.reshape(F.transpose(input_label_embeded, (1,0,2)), (batch_size, self.label_win_size *
                                                                                            self.insize)) # batch_size, win_size * D
                input_x_feature = F.reshape(F.transpose(input_x_feature, (1,0,2)), (batch_size, self.x_win_size * self.insize))

                all_input = F.concat((input_x_feature, input_label_embeded), axis=1) #shape = batch_size, (label_win + x_win)* insize
                all_input = self.mid_fc(all_input)  # batch_size, mid_size
                each_time_output = self.lstm(all_input)
                all_output.append(each_time_output)
            all_output = F.stack(all_output) # T, batch_size, out_size
            assert all_output.shape[0] == T
            all_output = F.transpose(all_output, (1, 0, 2))  # batch_size, T, out_size
            return list(F.separate(all_output, axis=0))  # list of T, out_size
        else: # at predict stage, cannot use labels information. but we can use only first win_size padded label to start
            labels = labels[:, 0:self.label_win_size, :]  # shape = B, win_size, D, which stores 0,1,0,0 binary array
            labels = xp.transpose(labels, (1,0,2)) # shape = win_size, B, D
            all_output = []
            for i in range(self.label_win_size):
                all_output.append(labels[i])

            for i in range(self.label_win_size, xs.shape[0]-self.x_win_size//2): # will produce T step
                previous_label = all_output[i-self.label_win_size: i]  # win_size, batch_size, out_size
                previous_label = F.transpose(F.stack(previous_label), (1,0,2)) # batch_size, win_size, out_size
                previous_label_embeded = self.make_label_embedding(previous_label)  # batch_size, win_size, D

                offset = np.concatenate([np.arange(-(self.x_win_size // 2), 0), np.arange(0, self.x_win_size // 2 + 1)])
                window_index = offset + i
                window_index = np.maximum(window_index, 0)
                assert window_index.shape[0] == self.x_win_size
                input_x_feature = xs[window_index]  # shape = win_size, B, D
                input_x_feature = F.reshape(F.transpose(input_x_feature, (1, 0, 2)),
                                            (batch_size, self.x_win_size * self.insize))
                input_label_embeded = F.reshape(previous_label_embeded, (batch_size, self.label_win_size *
                                                                                              self.insize))
                all_input = F.concat((input_x_feature, input_label_embeded),
                                     axis=1)  # shape = batch_size, (label_win + x_win)* insize
                all_input = self.mid_fc(all_input)  # batch_size, mid_size
                lstm_output = self.lstm(all_input)  # batch_size, out_size
                predict_label = (lstm_output.data > 0).astype(xp.int32)  # batch_size, out_size
                all_output.append(predict_label)
            del all_output[: self.label_win_size]
            all_output = xp.stack(all_output)  # T, batch_size, out_size
            assert all_output.shape[0] == T
            all_output = xp.transpose(all_output, (1, 0, 2))  # batch_size, T, out_size
            return xp.split(all_output,indices_or_sections=all_output.shape[0], axis=0)  # list of T, out_size



class LabelDependencyLayer(chainer.Chain):
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

    def __init__(self, database,  out_size:int,
                 train_mode=True, label_win_size=3,
                 x_win_size=1, label_dropout_ratio=0.4, use_space=True, use_temporal=True):
        super(LabelDependencyLayer, self).__init__()
        self.neg_pos_ratio = 3
        self.database = database
        self.out_size = out_size
        self.frame_node_num = config.BOX_NUM[self.database]
        self.use_space = use_space
        self.use_temporal = use_temporal
        with self.init_scope():
            if use_space:
                self.space_dependency_lstm = LabelDependencyLSTM(2048, 1024, self.out_size, label_win_size, x_win_size,
                                                          train_mode=train_mode, is_pad=True, dropout_ratio=label_dropout_ratio)
                self.space_flatten_fc = L.Linear(256 * 14 * 14, 2048)
            if use_temporal:
                self.temporal_dependency_lstm = LabelDependencyLSTM(2048, 1024, self.out_size,
                                                                    label_win_size=label_win_size,
                                                                    x_win_size=1,
                                                                    train_mode=train_mode, is_pad=True,
                                                                    dropout_ratio=label_dropout_ratio)

                self.temporal_flatten_fc = L.Linear(256 * 14 * 14, 2048)

            fuse_in_dim = 1024 * 2
            if (not use_space) or (not use_temporal):
                fuse_in_dim = 1024
            self.score_fc = L.Linear(fuse_in_dim, out_size)


    def forward(self, space_output, temporal_output, labels):  #  B, T, F, C'(256), H, W,  labels:B, T, F, class_num(12)
        '''
        :param xs: appearance features of all boxes feature across all frames
        :param gs:  geometry features of all polygons. each is 4 coordinates represent box
        :param crf_pact_structures: packaged graph structure contains supplementary information
        :return:
        '''
        if self.use_space:
            mini_batch, seq_len, box_num_frame, channel, height, width = space_output.shape
            assert channel * width * height == 256 * 14 * 14

            space_output = space_output.reshape(mini_batch * seq_len * box_num_frame, -1)
            space_output = self.space_flatten_fc(space_output)
            space_output = space_output.reshape(mini_batch * seq_len, box_num_frame, -1) # B, T, F, 2048
            space_output = list(F.separate(space_output, axis=0))  # list of F, 2048
            space_labels = list(F.separate(labels.reshape(mini_batch * seq_len, box_num_frame, -1), axis=0))  # list of F, 12
            space_output = self.space_dependency_lstm(space_output, space_labels)  # return list of F, 1024
            space_output = F.stack(space_output) # B*T, F, 1024
            space_output = space_output.reshape(space_output.shape[0] * space_output.shape[1], -1)  # B*T*F, 1024
        if self.use_temporal:
            mini_batch, seq_len, box_num_frame, channel, height, width = temporal_output.shape
            assert channel * width * height == 256 * 14 * 14
            temporal_output = F.transpose(temporal_output, (0, 2, 1, 3, 4, 5)) # B,F,T, C,H,W
            temporal_output = F.reshape(temporal_output, shape=(temporal_output.shape[0] * temporal_output.shape[1] * temporal_output.shape[2],
                                                                -1)) # B*F*T, C*H*W
            temporal_output = self.temporal_flatten_fc(temporal_output)  # B*F*T, 2048
            temporal_output = F.reshape(temporal_output, shape=(mini_batch*box_num_frame, seq_len, -1))  # B*F, T, 2048
            temporal_output = list(F.separate(temporal_output, axis=0))
            temporal_labels = F.transpose(labels, (0, 2, 1, 3))  # B, F, T, 12
            temporal_labels = list(F.separate(temporal_labels.reshape(mini_batch * box_num_frame, seq_len, -1), axis=0)) # list of T, 12

            temporal_output = self.temporal_dependency_lstm(temporal_output, temporal_labels)  # return list of T, 1024
            temporal_output = F.stack(temporal_output)  # B*F, T, 1024
            temporal_output = temporal_output.reshape(mini_batch, box_num_frame, seq_len, -1) # B, F, T, 1024
            temporal_output = F.transpose(temporal_output, (0,2,1,3)) # B, T, F, 1024
            assert temporal_output.shape[0] == mini_batch
            assert temporal_output.shape[1] == seq_len
            assert temporal_output.shape[2] == box_num_frame
            temporal_output = temporal_output.reshape(temporal_output.shape[0] * temporal_output.shape[1] * temporal_output.shape[2],-1)

        if self.use_temporal and self.use_space:
            fuse_in = F.concat([space_output, temporal_output], axis=1)
        elif self.use_space:
            fuse_in = space_output
        else:
            fuse_in = temporal_output

        return self.score_fc(fuse_in)   # B*T*F, 12


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


    def __call__(self, space_output, temporal_output, labels):
        # labels shape = B, T, F(9 or 8), 12
        # space_output/temporal_output shape =  B, T, F, C'(256), H, W, where F is box number in one frame image
        with chainer.cuda.get_device_from_array(labels) as device:
            predict = self.forward(space_output, temporal_output, labels)  # B*T*F,12s
            labels = self.xp.reshape(labels, (-1, self.out_size))
            pick_index, accuracy_pick_index = self.get_loss_index(predict, labels)
            loss = F.sigmoid_cross_entropy(predict[list(pick_index[0]), list(pick_index[1])],
                                           labels[list(pick_index[0]), list(pick_index[1])])
            accuracy = F.binary_accuracy(predict[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])
        return loss, accuracy


