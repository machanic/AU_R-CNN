import math

import chainer
import chainer.functions as F
import numpy as np
from time_axis_rcnn.constants.enum_type import TwoStreamMode
from time_axis_rcnn.model.time_segment_network.faster_rcnn_train_chain import _fast_rcnn_loc_loss

class Wrapper(chainer.Chain):

    def __init__(self, time_seg_train_chain_list, two_stream_mode):
        super(Wrapper, self).__init__()
        with self.init_scope():
            self.two_stream_mode = two_stream_mode
            if two_stream_mode == TwoStreamMode.rgb or two_stream_mode == TwoStreamMode.optical_flow:
                self.time_seg_train_chain = time_seg_train_chain_list[0]
            elif two_stream_mode == TwoStreamMode.rgb_flow:
                self.time_seg_train_chain_rgb = time_seg_train_chain_list[0]
                self.time_seg_train_chain_flow = time_seg_train_chain_list[1]


    def independent_bottom_process(self, train_chain, featuremap_1d, seg_info, gt_segments, anchor_spatial_scale):

        B, _, W = featuremap_1d.shape  # because W across mini-batch must be same, thus we need
        AU_group_id_arr = seg_info[:, 0]  # shape = (B,)
        features = train_chain.faster_extractor_backbone(featuremap_1d, AU_group_id_arr, W)
        rpn_locs, rpn_scores, anchor = train_chain.spn_module(
            features, AU_group_id_arr, W, anchor_spatial_scale)
        gt_rpn_loc, gt_rpn_label = train_chain.anchor_target_creator(
            gt_segments, anchor.reshape(-1, 2), W, seg_info)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs.reshape(-1, 2), gt_rpn_loc, gt_rpn_label, train_chain.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_scores.reshape(-1, 2), gt_rpn_label)
        return  features, rpn_locs, rpn_scores, rpn_loc_loss, rpn_cls_loss, anchor


    def shared_middle(self, batch_size, width_rgb, width_flow, rpn_scores_rgb, rpn_locs_rgb, rpn_scores_flow, rpn_locs_flow,
                      anchor_rgb, gt_segments_rgb, labels, seg_info):
        #  rpn_scores_rgb shape = (N, W_rgb * A, 2) rpn_scores_flow shape = (N, W_flow * A, 2)
        n_anchor = anchor_rgb.shape[1]
        rpn_locs_flow = F.transpose(rpn_locs_flow.reshape(batch_size, width_flow, n_anchor, 2), axes=(0, 3, 1, 2))  # (B, 2, W_flow, A)
        rpn_locs_flow = F.resize_images(rpn_locs_flow, (width_rgb, n_anchor))   # (B, 2, W_rgb, A)
        # B, W_rgb, A, 2 => B, W_rgb * A, 2
        rpn_locs_flow = F.reshape(F.transpose(rpn_locs_flow, axes=(0, 2, 3 ,1)), shape=(batch_size, width_rgb * n_anchor, 2))
        rpn_locs = F.average(F.stack([rpn_locs_rgb, rpn_locs_flow]), axis=0)

        rpn_scores_flow = F.transpose(rpn_scores_flow.reshape(batch_size, width_flow, n_anchor, 2), axes=(0, 3, 1, 2))
        rpn_scores_flow  = F.resize_images(rpn_scores_flow, (width_rgb, n_anchor)) # (B, 2, W_rgb, A)
        # B, W_rgb, A, 2 => B, W_rgb * A, 2
        rpn_scores_flow = F.reshape(F.transpose(rpn_scores_flow, axes=(0, 2, 3, 1)),
                                  shape=(batch_size, width_rgb * n_anchor, 2))
        rpn_scores = F.average(F.stack([rpn_scores_rgb,rpn_scores_flow]), axis=0)
        #  merge over!

        rois, roi_indices = self.time_seg_train_chain_rgb.nms_process(batch_size, width_rgb,
                                                                      n_anchor, rpn_scores, rpn_locs, anchor_rgb)

        sample_roi, sample_roi_index, gt_roi_loc, gt_roi_label = self.time_seg_train_chain_rgb.proposal_target_creator(
            rois, roi_indices, gt_segments_rgb, labels, seg_info,
            self.time_seg_train_chain_rgb.loc_normalize_mean, self.time_seg_train_chain_rgb.loc_normalize_std)
        return sample_roi, sample_roi_index, gt_roi_loc, gt_roi_label



    def independent_head(self, train_chain, features, sample_roi, sample_roi_index,spatial_scale,
                         gt_roi_label):

        sample_roi = sample_roi * spatial_scale
        roi_cls_loc, roi_score = train_chain.faster_head_module(
            features, sample_roi, sample_roi_index)

        n_sample = roi_cls_loc.shape[0]  # roi_cls_loc = (S, n_class *2), where S is sample RoI across all batch
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 2)  # shape = (S, n_class, 2), n_class是包含背景=0的
        roi_loc = roi_cls_loc[train_chain.xp.arange(n_sample), gt_roi_label]
        return roi_loc, roi_score


    def __call__(self, rgb_feature, flow_feature, gt_segments_rgb, gt_segments_flow, seg_info, labels):

        seg_info = chainer.cuda.to_cpu(seg_info)
        if self.two_stream_mode == TwoStreamMode.rgb:
            report = self.time_seg_train_chain(rgb_feature, seg_info, gt_segments_rgb, labels)
            chainer.reporter.report(report, self)
            return report['loss']
        elif self.two_stream_mode == TwoStreamMode.optical_flow:
            report = self.time_seg_train_chain(flow_feature, seg_info, gt_segments_flow, labels)
            chainer.reporter.report(report, self)
            return report['loss']
        elif self.two_stream_mode == TwoStreamMode.rgb_flow:
            if isinstance(gt_segments_rgb, chainer.Variable):
                gt_segments_rgb = gt_segments_rgb.data
            if isinstance(gt_segments_flow, chainer.Variable):
                gt_segments_flow = gt_segments_flow.data
            if isinstance(labels, chainer.Variable):
                labels = labels.data

            flow_scale = 1. / math.ceil(rgb_feature.shape[2] / flow_feature.shape[2])  # 0.1
            rgb_feature, rpn_locs_rgb, rpn_scores_rgb, rpn_loc_loss_rgb, rpn_cls_loss_rgb,anchor_rgb = \
                self.independent_bottom_process(self.time_seg_train_chain_rgb, rgb_feature, seg_info, gt_segments_rgb,
                                                1.0)

            flow_feature, rpn_locs_flow, rpn_scores_flow, rpn_loc_loss_flow, rpn_cls_loss_flow, anchor_flow = \
                self.independent_bottom_process(self.time_seg_train_chain_flow, flow_feature, seg_info, gt_segments_flow,
                                                flow_scale)

            batch_size, _, width_rgb = rgb_feature.shape
            batch_size, _, width_flow = flow_feature.shape
            sample_roi, sample_roi_index, gt_roi_loc, gt_roi_label =\
                self.shared_middle(batch_size, width_rgb, width_flow, rpn_scores_rgb, rpn_locs_rgb,
                                   rpn_scores_flow, rpn_locs_flow, anchor_rgb, gt_segments_rgb, labels, seg_info)

            roi_loc_rgb, roi_score_rgb = self.independent_head(self.time_seg_train_chain_rgb, rgb_feature, sample_roi,
                                                               sample_roi_index, 1.0, gt_roi_label)

            roi_loc_flow, roi_score_flow = self.independent_head(self.time_seg_train_chain_flow, flow_feature, sample_roi,
                                                                 sample_roi_index, flow_scale, gt_roi_label)


            roi_loc = F.average(F.stack([roi_loc_rgb, roi_loc_flow]), axis=0)
            roi_score = F.average(F.stack([roi_score_rgb, roi_score_flow]), axis=0)
            roi_loc_loss = _fast_rcnn_loc_loss(
                roi_loc, gt_roi_loc, gt_roi_label, self.time_seg_train_chain_rgb.roi_sigma)
            roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

            loss = rpn_loc_loss_rgb + rpn_cls_loss_rgb + rpn_loc_loss_flow + rpn_cls_loss_flow + roi_loc_loss + roi_cls_loss
            chainer.reporter.report({'rpn_loc_loss_rgb': rpn_loc_loss_rgb,
                                     'rpn_cls_loss_rgb': rpn_cls_loss_rgb,
                                     'rpn_loc_loss_flow': rpn_loc_loss_flow,
                                     'rpn_cls_loss_flow': rpn_cls_loss_flow,
                                     'roi_loc_loss': roi_loc_loss,
                                     'roi_cls_loss': roi_cls_loss,
                                     'loss': loss},
                                    self)
            return loss