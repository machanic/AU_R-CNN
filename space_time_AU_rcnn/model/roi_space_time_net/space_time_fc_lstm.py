import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from space_time_AU_rcnn.constants.enum_type import SpatialEdgeMode, TemporalEdgeMode


class SpaceTimeLSTM(chainer.Chain):

    def __init__(self, class_num, spatial_edge_mode: SpatialEdgeMode,
                 temporal_edge_mode: TemporalEdgeMode):
        super(SpaceTimeLSTM, self).__init__()
        self.class_num = class_num
        self.spatial_edge_mode = spatial_edge_mode
        self.temporal_edge_mode = temporal_edge_mode
        self.neg_pos_ratio = 3
        with self.init_scope():
            if temporal_edge_mode != TemporalEdgeMode.no_temporal:
                self.down_dim_fc_time = L.Linear(2048 * 7 * 7, 1024)
                self.temporal_fc_lstm = L.NStepLSTM(1, 1024, 1024, dropout=0)
            if spatial_edge_mode != SpatialEdgeMode.no_edge:
                self.down_dim_fc_space = L.Linear(2048 * 7 * 7, 1024)
                self.space_fc_lstm = L.NStepLSTM(1, 1024, 1024, dropout=0)
            # default used as space rnn
            self.score_fc = L.Linear(1024, class_num)



    def forward(self, xs):
        space_output = None
        temporal_output = None
        mini_batch, T, frame_box, *_ = xs.shape
        if self.temporal_edge_mode != TemporalEdgeMode.no_temporal:
            temporal_input = F.transpose(xs, axes=(0, 2, 1, 3, 4, 5))  # B, F, T, C, H, W
            temporal_input = F.reshape(temporal_input, shape=(mini_batch * frame_box * T, -1))
            temporal_input = self.down_dim_fc_time(temporal_input)   # B * F * T, 1024
            temporal_input = F.reshape(temporal_input, shape=(mini_batch * frame_box, T, -1))  # B * F, T, 1024
            temporal_input = list(F.separate(temporal_input, axis=0))
            temporal_output = F.stack(self.temporal_fc_lstm(None, None, temporal_input)[2])  # B*F, T, 1024
            # B, F, T, 1024
            temporal_output = F.reshape(temporal_output,
                                        shape=(xs.shape[0], xs.shape[2], xs.shape[1], 1024))
            temporal_output = F.transpose(temporal_output, axes=(0, 2, 1, 3))  # B, T, F, 1024
        if self.spatial_edge_mode != SpatialEdgeMode.no_edge:
            space_input = F.reshape(xs,
                                    shape=(mini_batch * T * frame_box, -1))

            space_input = self.down_dim_fc_space(space_input)  # B*T * F, 1024
            space_input = F.reshape(space_input, shape=(mini_batch * T, frame_box, -1))
            space_input = list(F.separate(space_input, axis=0))  # list of F, 1024
            space_output = F.stack(self.space_fc_lstm(None, None, space_input)[2])  # B*T, F, 1024

            space_output = F.reshape(space_output, shape=(xs.shape[0], xs.shape[1], xs.shape[2], -1)) # B, T, F, 1024

        if self.temporal_edge_mode!= TemporalEdgeMode.no_temporal and self.spatial_edge_mode!= SpatialEdgeMode.no_edge:
            assert space_output.shape == temporal_output.shape
            fusion_output = F.concat([space_output, temporal_output], axis=-1)

        elif self.spatial_edge_mode != SpatialEdgeMode.no_edge:
            fusion_output = space_output
        elif self.temporal_edge_mode != TemporalEdgeMode.no_temporal:
            fusion_output = temporal_output


        return fusion_output


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

    #mode = 1 : train, mode = 0: predict
    def __call__(self, xs, labels):  # xs shape = B,T,F,C,H,W, labels=  (batch, T, F, D)
        assert xs.ndim == 6
        assert labels.ndim == 4
        with chainer.cuda.get_device_from_array(xs.data) as device:
            fc_output = self.forward(xs) # B, T, F, 2048
            fc_output = F.reshape(fc_output, shape=(fc_output.shape[0] * fc_output.shape[1] * fc_output.shape[2], -1))
            fc_output = self.score_fc(fc_output)  # B * T * F, class_num
            labels = self.xp.reshape(labels, (-1, self.class_num))
            pick_index, accuracy_pick_index = self.get_loss_index(fc_output, labels)
            loss = F.sigmoid_cross_entropy(fc_output[list(pick_index[0]), list(pick_index[1])],
                                           labels[list(pick_index[0]), list(pick_index[1])])
            accuracy = F.binary_accuracy(fc_output[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])

            return loss, accuracy

    def predict(self, roi_features):  # B, T, F, C, H, W
        with chainer.cuda.get_device_from_array(roi_features.data) as device:
            fc_output = self.forward(roi_features)  # B, T, F, 2048
            mini_batch, seq_len, box_num, _ = fc_output.shape
            fc_output = F.reshape(fc_output, shape=(-1, 1024))
            fc_output = self.score_fc(fc_output)  # B * T * F, class_num
            pred = fc_output.reshape(mini_batch, seq_len, box_num, -1) # B, T, F, class_num
            pred = chainer.cuda.to_cpu(pred.data)  #  B, T, F, class_num
            pred = (pred > 0).astype(np.int32)
            return pred  # B, T, F, class_num
