import numpy as np

import chainer

import chainer.functions as F

from time_axis_rcnn.model.time_segment_network.anchor_target_creater import\
    AnchorTargetCreator

from time_axis_rcnn.model.time_segment_network.proposal_target_creater import\
    ProposalTargetCreator


class TimeSegmentRCNNTrainChain(chainer.Chain):

    """Calculate losses for Faster R-CNN and report them.

    This is used to train Faster R-CNN in the joint training scheme
    [#FRCNN]_.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    .. [#FRCNN] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        faster_rcnn (~chainercv.links.model.faster_rcnn.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#FRCNN]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#FRCNN]_.
        anchor_target_creator: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator_params: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.ProposalTargetCreator`.

    """

    def __init__(self, faster_extractor_backbone, faster_head_module, spn_module, rpn_sigma=3., roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator(), loc_normalize_mean=(0., 0.),
                 loc_normalize_std=(0.1, 0.2)):
        super(TimeSegmentRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.faster_head_module = faster_head_module
            self.faster_extractor_backbone = faster_extractor_backbone
            self.spn_module = spn_module
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std


    def nms_process(self, batch_size, width, n_anchor, rpn_scores, rpn_locs, anchor):
        rpn_fg_scores = \
            rpn_scores.reshape(batch_size, width, n_anchor, 2)[:, :, :, 1]  # 变成4维向量，取idx=1是前景的概率 shape = N, W, A
        rpn_fg_scores = rpn_fg_scores.reshape(batch_size, -1)  # n是batch_size，-1代表 ww x n_anchor个anchor

        rois = []
        roi_indices = []
        for i in range(batch_size):
            # rpn_loc[i].data 只是算的一个偏差，再结合anchor，才算出真正的roi位置
            # NMS算法
            roi = self.spn_module.proposal_layer(
                rpn_locs[i].data, rpn_fg_scores[i].data, anchor.reshape(-1, 2), width)  # 按照score从大到小排序，并且删掉超出屏幕的，以及他重叠很大IOU的被删除，即NMS算法
            # roi shape = R, 2
            batch_index = i * self.xp.ones((len(roi),), dtype=np.int32)  # 每张图下几个ROI全放1，再乘以图的index
            # batch——index仍旧是图片的index，而非ROI的
            rois.append(roi)
            roi_indices.append(batch_index)  # 是image的index

        rois = self.xp.concatenate(rois, axis=0)
        roi_indices = self.xp.concatenate(roi_indices, axis=0)
        return rois, roi_indices


    # 生成labels的代码要好好写写，比如段合并等等
    # 生成gt_segments也要好好写写
    def __call__(self, featuremap_1d, seg_info, gt_segments, labels):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            featuremap_1d (~chainer.Variable): A variable with a batch of 1d_featuremap.
                Its shape is :math:`(B, C, W)`, where W means width of timeline, B means AU group number
            seg_info (~chainer.Variable): shape = `(B, 2)` each row contains (AU group index, segment count of each batch index)
            gt_segments (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(B, R, 2)`. where R = config.MAX_SEGMENTS_PER_TIMELINE
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(B, R)`. The background is excluded from
                the definition.
            # NOTE that we delete scale argument in original ChainerCV
        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(gt_segments, chainer.Variable):
            gt_segments = gt_segments.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data

        B, _, W = featuremap_1d.shape  # because W across mini-batch must be same, thus we need
        AU_group_id_arr = seg_info[:, 0]  # shape = (B,)

        features = self.faster_extractor_backbone(featuremap_1d, AU_group_id_arr, W)

        # rpn_scores shape = (N, W * A, 2)
        # rpn_locs shape = (N, W * A, 2)
        # rois  = (R, 2), R 是跨越各个batch的，也就是跨越各个AU group的，每个AU group相当于独立的一张图片
        # roi_indices = (R,)
        # anchor shape =  (W, A, 2)

        rpn_locs, rpn_scores, anchor = self.spn_module(
            features, AU_group_id_arr, W, 1.0)
        rois, roi_indices = self.nms_process(B, W, anchor.shape[1], rpn_scores, rpn_locs, anchor)

        # Sample RoIs and forward，下面这句话才是1:3 sample
        # rois = (R, 2) roi_indices=(R,), gt_segments = (B, R', 2), label = (B, R'),
        sample_roi, sample_roi_index, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            rois, roi_indices, gt_segments, labels, seg_info,
            self.loc_normalize_mean, self.loc_normalize_std)
        assert sample_roi.shape[0] == sample_roi_index.shape[0] == gt_roi_loc.shape[0] == gt_roi_label.shape[0]
        # sample_roi = (S, 2), sample_roi_index=(S,), gt_roi_loc = (S, 2), gt_roi_label = (S)
        # S is across all batch index

        # roi_cls_loc = (S, class*2), roi_score = (S, class)
        roi_cls_loc, roi_score = self.faster_head_module(
            featuremap_1d, sample_roi, sample_roi_index)  # features shape = (B, C, W)

        # RPN losses, 根据anchor，为anchor打label，下面的这句话做按照pos:neg=0.5:0.5将gt_label修改
        # 返回gt_rpn_loc是一个偏差, shape = (B * W * A, 2); gt_rpn_label shape = (B * W * A,)
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            gt_segments, anchor.reshape(-1,2), W, seg_info)
        # gt_segments shape = (B,R,2), anchors = (W x A, 2)

        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs.reshape(-1,2), gt_rpn_loc, gt_rpn_label, self.rpn_sigma)

        # gt_rpn_label shape=(S',) ,each is 1/0/-1, indicate contain/ not contain object
        rpn_cls_loss = F.softmax_cross_entropy(rpn_scores.reshape(-1,2), gt_rpn_label)
        rpn_accuracy = F.accuracy(rpn_scores.reshape(-1,2), gt_rpn_label)
        # Losses for outputs of the head.
        # 每个分类回归一个坐标
        n_sample = roi_cls_loc.shape[0] #  roi_cls_loc = (S, n_class *2), where S is sample RoI across all batch
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 2) # shape = (S, n_class, 2), n_class是包含背景=0的
        # 由于gt_roi_label shape = (S, 12), 而非原始Faster RCNN的单label问题，所以修改如下:
        roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)

        roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)  # multi-label 问题分类用sigmoid cross entropy
        accuracy = F.accuracy(roi_score, gt_roi_label)
        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        report = {'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'roi_loc_loss': roi_loc_loss,
                                 'roi_cls_loss': roi_cls_loss,
                                 'accuracy': accuracy,
                                'rpn_accuracy':rpn_accuracy,
                                 'loss': loss}
        return report


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    return F.sum(y)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)
    in_weight = xp.zeros_like(gt_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[gt_label > 0] = 1  # gt_label 是RPN返回的1或者0的label
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= xp.sum(gt_label >= 0)
    return loc_loss
