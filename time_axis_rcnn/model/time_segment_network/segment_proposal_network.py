import chainer
from collections import defaultdict
import numpy as np
import config
from time_axis_rcnn.model.time_segment_network.dilated_convolution_1d import DilatedConvolution1D
from time_axis_rcnn.model.time_segment_network.proposal_creater import ProposalCreator
from time_axis_rcnn.model.time_segment_network.generate_anchors import get_all_anchors
import chainer.functions as F

class SegmentProposalNetwork(chainer.Chain):

    def __init__(self, database, mid_channels, n_anchors=len(config.ANCHOR_SIZE),
                 proposal_creator_params={},
                 initialW=None):
        super(SegmentProposalNetwork, self).__init__()
        self.proposal_layer = ProposalCreator(**proposal_creator_params)
        self.groups = len(config.BOX_NUM[database])
        self.conv_layers = defaultdict(list)
        self.n_anchors = n_anchors
        with self.init_scope():
            # 下面的类同样要加group
            # score's channel 2 means 1/0  loc's channel 2 means x_min, x_max
            self.score_layers = {}
            self.loc_layers = {}
            for group_id in range(self.groups):
                setattr(self, "score_#{0}".format(group_id), DilatedConvolution1D(mid_channels, n_anchors * 2, 1, 1, 0,
                                                                                  dilate=1, initialW=initialW))
                self.score_layers[group_id] = "score_#{0}".format(group_id)
                setattr(self, "loc_#{0}".format(group_id), DilatedConvolution1D(mid_channels, n_anchors * 2, 1, 1, 0,
                                                                                  dilate=1, initialW=initialW))
                self.loc_layers[group_id] = "loc_#{0}".format(group_id)


    def __call__(self, x, AU_group_id_array, seq_len):
        n, _, ww = x.shape # Note that n is number of AU groups
        anchor = get_all_anchors(seq_len, stride=config.ANCHOR_STRIDE, sizes=config.ANCHOR_SIZE)  # W, A, 2
        n_anchor = anchor.shape[1]
        assert n_anchor == self.n_anchors

        score_out_list = []
        loc_out_list = []
        for batch_idx, group_id in enumerate(AU_group_id_array):
            x_inside_batch = F.expand_dims(x[batch_idx], axis=0) # 1,C,W
            score_out = getattr(self, self.score_layers[group_id])(x_inside_batch)  # 1, n_anchors * 2, w
            loc_out = getattr(self, self.loc_layers[group_id])(x_inside_batch)  # 1, n_anchors * 2, w
            score_out_list.append(score_out)  #
            loc_out_list.append(loc_out)
        rpn_locs = F.concat(loc_out_list, axis=0) # shape = B, C, W; output channel is n_anchor * 2
        # 1. transpose to (B, W, A * 2) then reshape to (B, W * A, 2)
        rpn_locs = rpn_locs.transpose((0, 2, 1)).reshape(n, -1,
                                                            2)  # put channel last dimension, then reshape to (B, W * A, 2) , 第二个维度每个都是一个anchor

        rpn_scores = F.concat(score_out_list, axis=0)  # output channel is n_anchor * 2
        rpn_scores = rpn_scores.transpose(0, 2, 1)  # put channel last dimension shape = (N, W, C) C= A * 2
        rpn_fg_scores = \
            rpn_scores.reshape(n, ww, n_anchor, 2)[:, :, :, 1]  # 变成4维向量，取idx=1是前景的概率 shape = N, W, A
        rpn_fg_scores = rpn_fg_scores.reshape(n, -1)  # n是batch_size，-1代表 ww x n_anchor个anchor
        rpn_scores = rpn_scores.reshape(n, -1, 2)  # shape (N, W * A, 2)

        rois = []
        roi_indices = []
        for i in range(n):
            # rpn_loc[i].data 只是算的一个偏差，再结合anchor，才算出真正的roi位置
            # NMS算法
            roi = self.proposal_layer(
                rpn_locs[i].data, rpn_fg_scores[i].data, anchor.reshape(-1, 2), ww)  # 按照score从大到小排序，并且删掉超出屏幕的，以及他重叠很大IOU的被删除，即NMS算法
            # roi shape = R, 2
            batch_index = i * self.xp.ones((len(roi),), dtype=np.int32)  # 每张图下几个ROI全放1，再乘以图的index
            # batch——index仍旧是图片的index，而非ROI的
            rois.append(roi)
            roi_indices.append(batch_index)  # 是image的index

        rois = self.xp.concatenate(rois, axis=0)
        roi_indices = self.xp.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor  # 因为rpn_scores,所以RPN的loss是前景背景都要算
