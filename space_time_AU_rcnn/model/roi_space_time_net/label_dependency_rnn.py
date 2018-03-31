from collections import defaultdict


import config
import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np

from space_time_AU_rcnn.model.roi_space_time_net.attention_base_block import MultiHeadAttention


class BiDirectionLabelDependencyRNN(chainer.Chain):
    def __init__(self, insize, outsize, class_num, label_win_size=3, x_win_size=1, train_mode=True, is_pad=True,
                 dropout_ratio=0.4):
        super(BiDirectionLabelDependencyRNN, self).__init__()
        mid_size = 512
        self.out_size = outsize
        self.mid_size = mid_size
        with self.init_scope():
            self.forward_ld_rnn = LabelDependencyRNN(insize, mid_size, class_num, label_win_size,
                                                     x_win_size, train_mode, is_pad, dropout_ratio=dropout_ratio)
            self.backward_ld_rnn = LabelDependencyRNN(insize, mid_size, class_num, label_win_size,
                                                      x_win_size, train_mode, is_pad, dropout_ratio=dropout_ratio)
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


class LabelDependencyRNN(chainer.Chain):
    # if we set pad=False, the returned value of axis T is smaller than input Variable
    def __init__(self, insize, outsize, class_num, label_win_size=3, x_win_size=1, train_mode=True, is_pad=True, dropout_ratio=0.4):
        super(LabelDependencyRNN, self).__init__()
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
            self.out_fc = L.Linear(self.mid_size, outsize)



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
        # self.clear()
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
                each_time_output = self.out_fc(all_input)
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
                lstm_output = self.out_fc(all_input)  # batch_size, out_size
                predict_label = (lstm_output.data > 0).astype(xp.int32)  # batch_size, out_size
                all_output.append(predict_label)
            del all_output[: self.label_win_size]
            all_output = xp.stack(all_output)  # T, batch_size, out_size
            assert all_output.shape[0] == T
            all_output = xp.transpose(all_output, (1, 0, 2))  # batch_size, T, out_size
            return xp.split(all_output,indices_or_sections=all_output.shape[0], axis=0)  # list of T, out_size



class LabelDependencyRNNLayer(chainer.Chain):
    def __init__(self, database, in_size, class_num, train_mode=True, label_win_size=1, ld_dropout=0.1):
        super(LabelDependencyRNNLayer, self).__init__()
        self.frame_node_num = config.BOX_NUM[database]
        self.class_num = class_num
        self.neg_pos_ratio = 3
        with self.init_scope():
            self.label_dep_rnn = LabelDependencyRNN(in_size, class_num, class_num, label_win_size=label_win_size,
                                                    x_win_size=1,
                                                    train_mode=train_mode, is_pad=True, dropout_ratio=ld_dropout)

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


    def __call__(self, xs, labels): # B, T, F, 2048, labels: B, T, F, 12
        xs = F.transpose(xs, (0, 2, 1, 3)) # B, F, T, 2048
        orig_labels = labels
        labels = F.transpose(labels, (0,2,1,3)) # B, F, T, 12
        mini_batch, frame_node, T, _ = xs.shape
        xs = xs.reshape(xs.shape[0] * xs.shape[1], xs.shape[2], xs.shape[3])
        labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3])
        xs = list(F.separate(xs, axis=0))  # list of T, 2048
        labels = list(F.separate(labels, axis=0)) # list of T, 12
        output = F.stack(self.label_dep_rnn(xs, labels)) # B * F, T, 12
        output = output.reshape(mini_batch, frame_node, T, -1)
        output = F.transpose(output, (0, 2, 1, 3)) # B, T, F, D
        output = output.reshape(-1, self.class_num) # B * T * F, 12
        orig_labels = orig_labels.reshape(-1, self.class_num)
        assert output.shape == orig_labels.shape
        pick_index, accuracy_pick_index = self.get_loss_index(output, orig_labels)
        loss = F.sigmoid_cross_entropy(output[list(pick_index[0]), list(pick_index[1])],
                                       orig_labels[list(pick_index[0]), list(pick_index[1])])
        accuracy = F.binary_accuracy(output[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                     orig_labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])

        return loss, accuracy


class LabelDependencyLayer(chainer.Chain):

    def __init__(self, database,  in_size:int, class_num:int,
                 train_mode=True, label_win_size=1):
        super(LabelDependencyLayer, self).__init__()
        self.neg_pos_ratio = 3
        self.database = database
        self.frame_node_num = config.BOX_NUM[self.database]
        self.train_mode = train_mode
        self.label_win_size = label_win_size
        self.class_num = class_num
        with self.init_scope():
            # self.label_attention = MultiHeadAttention(n_heads=n_head, d_model=in_size, d_k=in_size//n_head, d_v=in_size//n_head,
            #                                           dropout=label_dropout_ratio)
            self.label_embed = L.EmbedID(class_num + 1, in_size, ignore_label=-1,
                                         initialW=I.Uniform(1. / in_size))
            self.score_fc_1 = L.Linear(in_size * (label_win_size + 1), 1024)  # last predict layer
            self.score_fc_2 = L.Linear(1024, class_num)

    def make_label_embedding(self, batch_labels):
        # labels shape = (batch, N, class_number)
        # out shape = (batch, N, embed_length)
        xp = chainer.cuda.cupy.get_array_module(batch_labels)
        batch_size, N, class_number = batch_labels.shape
        batch_embeded = []
        lengths = []
        label_dict = defaultdict(dict)  # all_label_id position -> labels index
        batch_label_id_list = []
        for i in range(batch_size):
            labels = batch_labels[i]
            each_label_id_list = []
            batch_length = 0
            for label_index, label in enumerate(labels):
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


    def forward_train(self, appearance_features, labels):  #  appearance_features: B, T, F, 2048,  labels: B, T, F, class_num(12)
        '''
        :param appearance_features: appearance features of all boxes feature across all frames
        :param labels:  labels of each box cross frames
        :return:
        '''
        with chainer.cuda.get_device_from_array(appearance_features.data) as device:

            appearance_features = F.transpose(appearance_features, axes=(0, 2, 1, 3)) # B, F, T, 2048
            orig_labels = labels
            # labels = labels.transpose((0, 2, 1, 3))  # B, F, T, 12
            mini_batch, T, box_num_frame, class_num = labels.shape
            assert box_num_frame == self.frame_node_num
            assert T == appearance_features.shape[2]
            assert mini_batch == appearance_features.shape[0]
            labels = labels.reshape(labels.shape[0], labels.shape[1] * labels.shape[2], labels.shape[3]) # B, T * F, 12
            labels_embeded = self.make_label_embedding(labels)  # B, T * F, d_k
            labels_embeded = labels_embeded.reshape(mini_batch, T, box_num_frame, labels_embeded.shape[-1])  # B, T, F, d_k
            assert T > 0
            fuse_information = []
            fuse_information.append(F.tile(appearance_features[:, :, 0, :], reps=(1,1, self.label_win_size + 1)))  # B, F, 4096
            for t in range(1, T):
                # labels_embeded_previous = labels_embeded[:, 0:t, :, :]  # B, t, F, d_model
                if t - self.label_win_size < 0:
                    labels_embeded_previous = labels_embeded[:, :t, :, :] # B, t, F, d_model
                    labels_embeded_previous = F.pad(labels_embeded_previous, ((0,0), (self.label_win_size - t, 0), (0,0), (0,0)), mode="edge")
                else:
                    labels_embeded_previous = labels_embeded[:, t-self.label_win_size: t, :, : ]  # B, win, F, d_model
                labels_embeded_previous = F.transpose(labels_embeded_previous, (0, 2, 1, 3))  # B, F, win, d_model
                labels_embeded_previous = labels_embeded_previous.reshape(labels_embeded_previous.shape[0],
                        labels_embeded_previous.shape[1], labels_embeded_previous.shape[2] * labels_embeded_previous.shape[3])  # B, F, win*d_model
                # labels_embeded_previous = labels_embeded_previous.reshape(mini_batch, t * box_num_frame, -1) # B, t * F, d_model
                appearance_features_t = appearance_features[:, :, t, :] # B, F, 2048
                # fuse_information_t = self.label_attention(labels_embeded_previous_t, labels_embeded_previous, labels_embeded_previous)  # B, F, 2048
                fuse_information.append(F.concat((appearance_features_t, labels_embeded_previous), axis=2))  # B, F, 2048 * (win+1)

            fuse_information = F.stack(fuse_information, axis=1)   # B, T, F, 4096
            fuse_information = fuse_information.reshape(-1, fuse_information.shape[-1]) # B * T * F, 4096
            score = self.score_fc_1(fuse_information)
            score = self.score_fc_2(score)
            orig_labels = self.xp.reshape(orig_labels, (-1, self.class_num))
            pick_index, accuracy_pick_index = self.get_loss_index(score, orig_labels)
            loss = F.sigmoid_cross_entropy(score[list(pick_index[0]), list(pick_index[1])],
                                           orig_labels[list(pick_index[0]), list(pick_index[1])])
            accuracy = F.binary_accuracy(score[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         orig_labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])

            return loss, accuracy

    def forward_test(self, appearance_features):  # appearance_features shape = B, T, F, 2048

        appearance_features = F.transpose(appearance_features, axes=(0, 2, 1, 3))  # B, F, T, 2048
        mini_batch, box_num_frame, T, _ = appearance_features.shape
        first_fuse_information = F.tile(appearance_features[:, :, 0, :], reps=(1, 1, self.label_win_size + 1 )) # B, F, 4096
        first_fuse_information = first_fuse_information.reshape(first_fuse_information.shape[0] * \
                                                                first_fuse_information.shape[1], -1)

        predict_score_0 = self.score_fc_1(first_fuse_information)
        predict_score_0 = self.score_fc_2(predict_score_0)  # B * F, 12
        predict_label_list = [(predict_score_0.data > 0).astype(self.xp.int32)]
        fuse_information = []
        for t in range(1, T):
            if t - self.label_win_size < 0:
                label_win_size = t
            else:
                label_win_size = -self.label_win_size
            predict_labels = F.stack(predict_label_list[-label_win_size : ])  # win, B*F, 12
            if t - self.label_win_size < 0:
                predict_labels = F.tile(predict_labels, reps=(self.label_win_size - t, 1, 1))
            predict_labels = predict_labels.reshape(self.label_win_size, mini_batch, box_num_frame, self.class_num)
            predict_labels = F.transpose(predict_labels, (1, 0, 2, 3)) # B, win, F, 12
            predict_labels = predict_labels.reshape(predict_labels.shape[0], predict_labels.shape[1] * predict_labels.shape[2], predict_labels.shape[3]) # B, win * F, 12
            labels_embeded_previous = self.make_label_embedding(predict_labels) # B, win * F, d_model
            labels_embeded_previous =  F.transpose(labels_embeded_previous.reshape(mini_batch, self.label_win_size, box_num_frame, -1), (0, 2, 1, 3)) # B, F, win, d_model
            labels_embeded_previous = labels_embeded_previous.reshape(mini_batch, box_num_frame, self.label_win_size * labels_embeded_previous.shape[-1])  # B, F, win*d_model
            appearance_features_t = appearance_features[:, :, t, :] # B, F, 2048
            # fuse_information_t = self.label_attention(present_labels_t, labels_embeded_previous, labels_embeded_previous)
            concat_fuse = F.concat((appearance_features_t, labels_embeded_previous), axis=2)   # B, F, 4096
            fuse_information.append(concat_fuse)
            concat_fuse = concat_fuse.reshape(concat_fuse.shape[0] * concat_fuse.shape[1], -1)  # B * F, 4096
            predict_score = self.score_fc_1(concat_fuse)
            predict_score = self.score_fc_2(predict_score)  # B * F, 12
            predict_label_list.append((predict_score.data > 0).astype(self.xp.int32))
        predict_labels = F.stack(predict_label_list, 1) # B*F, T, d_k
        predict_labels = F.reshape(predict_labels, (mini_batch, box_num_frame, T, self.class_num)) # B, F, T, C
        predict_labels = F.transpose(predict_labels, axes=(0, 2, 1, 3))  # B, T, F, C
        return predict_labels



    def __call__(self, appearance_features, labels):
        # labels shape = B, T, F(9 or 8), 12
        # space_output/temporal_output shape =  B, T, F, C'(256), H, W, where F is box number in one frame image
        with chainer.cuda.get_device_from_array(labels) as device:
            if self.train_mode:
                return self.forward_train(appearance_features, labels)  # loss, accuracy
            return self.forward_test(appearance_features, labels)
    def predict(self, appearance_features):
        with chainer.cuda.get_device_from_array(appearance_features) as device:
            return self.forward_test(appearance_features)
