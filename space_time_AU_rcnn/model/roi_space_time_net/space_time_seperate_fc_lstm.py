import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import config
from space_time_AU_rcnn.constants.enum_type import SpatialEdgeMode, TemporalEdgeMode

class SpaceTimeSepFcLSTM(chainer.Chain):

    def __init__(self, database, class_num, spatial_edge_mode: SpatialEdgeMode,
                 temporal_edge_mode: TemporalEdgeMode):
        super(SpaceTimeSepFcLSTM, self).__init__()
        self.class_num = class_num
        self.neg_pos_ratio = 3
        self.spatial_edge_mode = spatial_edge_mode
        self.temporal_edge_mode = temporal_edge_mode
        self.database = database
        self.frame_box_num = config.BOX_NUM[self.database]
        with self.init_scope():
            if temporal_edge_mode != TemporalEdgeMode.no_temporal:
                for i in range(self.frame_box_num):
                    setattr(self, "temporal_lstm_{}".format(i),L.NStepLSTM(1, 2048, 512, dropout=0.1))

            if spatial_edge_mode != SpatialEdgeMode.no_edge:
                self.space_fc_lstm = L.NStepLSTM(1, 2048, 512, dropout=0.1)

            self.score_fc = L.Linear(512, class_num)



    def forward(self, xs):
        space_output = None
        temporal_output = None
        if self.temporal_edge_mode != TemporalEdgeMode.no_temporal:
            temporal_input = F.transpose(xs, axes=(0, 2, 1, 3))  # B, F, T, D
            assert temporal_input.shape[1] == config.BOX_NUM[self.database]
            all_temporal_output = []
            for idx, temporal_input_each_box in enumerate(F.separate(temporal_input, axis=1)):  # B,F,T,D =>F list of B, T, D
                # temporal_input_each_box : list of (T,D)
                temporal_input_each_box = list(F.separate(temporal_input_each_box, axis=0)) # list of (T,D)
                _, _, temporal_output = getattr(self, "temporal_lstm_{}".format(idx))(None, None, temporal_input_each_box)
                temporal_output = F.stack(temporal_output)  # B,T,D
                all_temporal_output.append(temporal_output)
            all_temporal_output = F.stack(all_temporal_output, axis=1)# B,F,T,D
            temporal_output = F.transpose(all_temporal_output, axes=(0,2,1,3)) # B,T,F,D

        if self.spatial_edge_mode != SpatialEdgeMode.no_edge:  # B,T,F,D
            minibatch, T, frame_box, _ = xs.shape  # B,T,F,D
            space_input = F.reshape(xs,
                                    shape=(xs.shape[0] * xs.shape[1], xs.shape[2], xs.shape[3]))  # B*T,F,D
            space_input = list(F.separate(space_input, axis=0)) # list of F,D
            _, _, space_output = self.space_fc_lstm(None, None, space_input)  # B*T, F, 1024
            # B, T, F, D
            space_output = F.stack(space_output) # B*T, F, 1024
            space_output = F.reshape(space_output, shape=(minibatch, T, frame_box, -1))

        if self.temporal_edge_mode!= TemporalEdgeMode.no_temporal and self.spatial_edge_mode!= SpatialEdgeMode.no_edge:
            assert space_output.shape == temporal_output.shape
            fusion_output = F.concat([space_output, temporal_output], axis=3)

        elif self.spatial_edge_mode != SpatialEdgeMode.no_edge:
            fusion_output = space_output
        elif self.temporal_edge_mode != TemporalEdgeMode.no_temporal:
            fusion_output = temporal_output
        fc_input = F.reshape(fusion_output,
                                   shape=(fusion_output.shape[0] * fusion_output.shape[1] * fusion_output.shape[2],
                                          -1))
        score = self.score_fc(fc_input)
        return F.reshape(score, shape=(fusion_output.shape[0], fusion_output.shape[1], fusion_output.shape[2], -1))


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
        assert xs.ndim == 4
        assert labels.ndim == 4
        with chainer.cuda.get_device_from_array(xs.data) as device:
            fc_output = self.forward(xs) # B, T, F, 2048
            fc_output = F.reshape(fc_output, (-1, self.class_num))
            labels = self.xp.reshape(labels, (-1, self.class_num))
            pick_index, accuracy_pick_index = self.get_loss_index(fc_output, labels)
            loss = F.sigmoid_cross_entropy(fc_output[list(pick_index[0]), list(pick_index[1])],
                                           labels[list(pick_index[0]), list(pick_index[1])])
            accuracy = F.binary_accuracy(fc_output[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         labels[[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])]])

            return loss, accuracy

    def predict(self, roi_features):  # B, T, F,D
        with chainer.cuda.get_device_from_array(roi_features.data) as device:
            pred = self.forward(roi_features)  # B, T, F, 12
            pred = chainer.cuda.to_cpu(pred.data)  #  B, T, F, class_num
            pred = (pred > 0).astype(np.int32)
            return pred  # B, T, F, class_num
