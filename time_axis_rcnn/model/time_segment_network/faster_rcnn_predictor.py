# Mofidied work:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# --------------------------------------------------------
#
# Original works by:
# --------------------------------------------------------
# Faster R-CNN implementation by Chainer
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/mitmul/chainer-faster-rcnn
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

from __future__ import division

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from time_axis_rcnn.model.time_segment_network.faster_rcnn_train_chain import TimeSegmentRCNNTrainChain
from time_axis_rcnn.model.time_segment_network.util.bbox.bbox_util import decode_segment_target
from time_axis_rcnn.model.time_segment_network.util.bbox.non_maximum_suppression import non_maximum_suppression
from collections import defaultdict


class TimeSegmentRCNNPredictor(chainer.Chain):

    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`chainer.Chain` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two func :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :func:`FasterRCNN.predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (callable Chain): A callable that takes a BCHW image
            array and returns feature maps.
        rpn (callable Chain): A callable that has the same interface as
            :class:`chainercv.links.RegionProposalNetwork`. Please refer to
            the documentation found there.
        head (callable Chain): A callable that takes
            a BCHW array, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`prepare`.
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please
            refer to a docstring found for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(
            self,feature_backbone, spn, head,
            loc_normalize_mean=(0., 0.),
            loc_normalize_std=(0.1, 0.2),
    ):
        super(TimeSegmentRCNNPredictor, self).__init__()
        with self.init_scope():
            self.train_chain = TimeSegmentRCNNTrainChain(feature_backbone, head, spn)
            self.feature_backbone = self.train_chain.faster_extractor_backbone
            self.spn = self.train_chain.spn_module
            self.head = self.train_chain.faster_head_module

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset("evaluate")


    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    # only used by predict
    def __call__(self, x):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (~chainer.Variable): 1D feature image variable. (B,C,W)

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 2)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 2)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        batch, channel, ww = x.shape
        # x, AU_group_id_array, seq_len, scale=1.
        h = self.feature_backbone(x)
        # rpn_scores shape = (N, W * A, 2)
        # rpn_locs shape = (N, W * A, 2)
        # rois  = (R, 2), R 是跨越各个batch的，也就是跨越各个AU group的，每个AU group相当于独立的一张图片
        # roi_indices = (R,)
        # anchor shape =  (W, A, 2)
        rpn_locs, rpn_scores, anchor = self.spn(h, 1.0)
        rois, roi_indices = self.train_chain.nms_process(batch, ww, anchor.shape[1], rpn_scores, rpn_locs, anchor)
        roi_cls_locs, roi_scores = self.head(
            x, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

        # roi_cls_loc = (S, class*2), roi_score = (S, class), rois = (R, 2), roi_indices=(R,)

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_score):
        # raw_cls_bbox = R, class * 2; raw_score = R, class
        segments = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_seg_l = raw_cls_bbox.reshape(-1, self.n_class, 2)[:, l, :] # R, 2
            prob_l = raw_score[:, l] # shape = R, raw_score is output of sigmoid function value
            mask = prob_l > self.score_thresh  # R
            if np.all(np.logical_not(cuda.to_cpu(mask))):
                continue
            cls_seg_l = cls_seg_l[mask]  # R', 2,   R' 有可能是0
            prob_l = prob_l[mask]  # R'
            keep = non_maximum_suppression(
                cls_seg_l, self.nms_thresh, prob_l) # 每个分类内部NMS
            segments.append(cuda.to_cpu(cls_seg_l[keep]))
            score.append(cuda.to_cpu(prob_l[keep]))
            # The labels are in [0, self.n_class - 2]. 抛去完全背景的1个label
            label.append((l - 1) * np.ones((len(keep),)))
        if segments:
            segments = np.concatenate(segments, axis=0).astype(np.float32)  # R', 2
            label = np.concatenate(label, axis=0).astype(np.int32)  # R'
            score = np.concatenate(score, axis=0).astype(np.float32)  # R'
        else:
            segments = np.array([])
            label = np.array([])
            score = np.array([])
        return segments, label, score

    def predict(self, featuremap_1D):
        """Detect segments from timeline.

        This method predicts objects for each image.

        Args:
            featuremap_1D (iterable of numpy.ndarray): Arrays holding pre-extracted features. shape=(B, C, W)

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **segments**: A list of float arrays of shape :math:`(R, 2)`, \
                the list length is batch size
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :obj:`(x_min, x_max)` in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R, class_number)`. \
               Each value indicates the class of the bounding box. .
           * **scores** : A list of float arrays of shape :math:`(R, class_number)`. \
               Each value indicates how confident the prediction is.

        """

        segment_list = list()
        labels = list()
        scores = list()
        seq_len = featuremap_1D.shape[2]
        for feature_inside_batch in featuremap_1D:
            with chainer.function.no_backprop_mode():
                x_var = F.expand_dims(feature_inside_batch, axis=0)
                # roi_cls_loc = (R, class*2), roi_score = (R, class), rois = (R, 2), roi_indices=(R,)
                roi_cls_locs, roi_scores, rois, _ = self.__call__(
                    x_var)
                assert roi_cls_locs.shape[0] == rois.shape[0]
            # We are assuming that batch size is 1.
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi = rois # shape = (R, 2) ,R across all batch index

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean),
                                self.n_class)  # shape = (n_class * 2)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std),
                               self.n_class) # shape = (n_class * 2)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)  # (R, class * 2)  element-wise dot (class*2) = (R, class* 2)
            roi_cls_loc = roi_cls_loc.reshape(-1, self.n_class, 2)  # shape = (R, class, 2)
            # roi (R, 1, 2) to (R, class, 2), 类似tile复制
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)  # R, class, 2
            cls_bbox = decode_segment_target(roi.reshape(-1, 2), roi_cls_loc.reshape(-1, 2))

            cls_bbox = cls_bbox.reshape(-1, self.n_class * 2) # R, class * 2
            # clip bounding box
            cls_bbox = self.xp.clip(
                cls_bbox, 0, seq_len)  # 开眼了

            prob = F.softmax(roi_score).data
            raw_cls_bbox = cuda.to_gpu(cls_bbox)  # R, class * 2
            raw_prob = cuda.to_gpu(prob)  # R, class
            # the number of foreground ROIs are constant until they are NMSed
            segments, label, score = self._suppress(raw_cls_bbox, raw_prob)
            segment_list.append(segments)
            labels.append(label)
            scores.append(score)

        return segment_list, labels, scores  # list of (R,2) list of (R,) list of R
