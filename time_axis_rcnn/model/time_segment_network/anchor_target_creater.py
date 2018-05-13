import numpy as np

import chainer
from chainer import cuda
from time_axis_rcnn.model.time_segment_network.util.bbox.bbox_util import encode_segment_target
from time_axis_rcnn.model.time_segment_network.util.bbox.bbox_util import segments_iou


# 为了计算RPN losses使用的anchor的gt label, 根据anchor，为anchor打label
class AnchorTargetCreator(object):

    """Assign the ground truth segments to anchors.

    Assigns the ground truth segments to anchors for training Segment
    Proposal Networks.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :obj:`lstm_end_to_end.util.bbox.bbox_iou.bbox2loc`.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.3):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, gt_segments, anchor, sequence_length, seg_info):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            gt_segments (array): Coordinates of bounding boxes. Its shape is
                :math:`(B, R, 2)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(K*A, 2)`.
            sequence_length  int: A tuple :obj:`W`, which
                is length of video frames.
            seg_info (np.array):  (B, 2) columns : AU group id and gt_box_number per timeline

        Returns:
            (array, array):

            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """
        xp = cuda.get_array_module(gt_segments)
        gt_segments = cuda.to_cpu(gt_segments)
        anchor = cuda.to_cpu(anchor)
        seg_info = cuda.to_cpu(seg_info)

        mini_batch_size = gt_segments.shape[0]

        n_anchor = len(anchor)  # W x A
        inside_index = _get_inside_index(anchor, sequence_length)
        anchor = anchor[inside_index]   # inside_num, 2  # 这就意味着，每个batch传入的sequence_length必须一样长
        batch_labels = []
        batch_locs = []
        for b_id in range(mini_batch_size):
            _, gt_seg_num = seg_info[b_id]
            gt_seg = gt_segments[b_id][:gt_seg_num]
            argmax_ious, label = self._create_label(
                inside_index, anchor, gt_seg)  # 这个label指的是1或者0，有没有物体在里面的label
            # compute bounding box regression targets, argmax_ious指的是每个anchor所对的最大IOU的gt_seg的index
            loc = encode_segment_target(anchor, gt_seg[argmax_ious, :])  # shape = R, 2; 编码偏差

            # map up to original set of anchors, inside_index让长度缩减了，搞回原始长度 W x A
            label = _unmap(label, n_anchor, inside_index, fill=-1)
            loc = _unmap(loc, n_anchor, inside_index, fill=0)
            if xp != np:
                loc = chainer.cuda.to_gpu(loc)
                label = chainer.cuda.to_gpu(label)
            batch_labels.append(label)
            batch_locs.append(loc)

        batch_locs = xp.concatenate(batch_locs, axis=0)  # shape = S, 2;  S is all segments' ground truth segment number across batch
        batch_labels = xp.concatenate(batch_labels, axis=0)  # shape = S; S is all segments' ground truth segment number
        return batch_locs, batch_labels

    def _create_label(self, inside_index, anchor, gt_seg):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index), ), dtype=np.int32) # label指得是anchor的label，所以长度与anchor个数匹配
        label.fill(-1)

        # argmax_ious是anchor所对的最大iou的gt segments index; max_ious是这些gt segments的iou;
        # gt_argmax_ious是每个gt_segments所对应的最大anchor ious的index
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, gt_seg, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1  # 这些anchor都是可以被设置为1的

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)  # n_sample is the number of regions to produce
        pos_index = np.where(label == 1)[0]  # 找出anchor的index
        if len(pos_index) > n_pos: # 如果label=1太多了，从label=1的里面随机挑变成0
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, segments, inside_index):
        # ious between the anchors and the gt segments
        ious = segments_iou(anchor, segments)  # anchor shape = N, 2;  segments shape = K, 2; return (N, K)
        argmax_ious = ious.argmax(axis=1)  # 挑选gt segments的index
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]  # 取出这些gt segments的iou，shape= N = inside_index
        gt_argmax_ious = ious.argmax(axis=0)  # 挑选anchor, 对每个gt,从纵向看，挑出一个最大的行,shape= K
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # 仅仅为了下一句
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # 挑出anchor的行号，所以用[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, seq_len):
    # Calc indicies of anchors which are located completely inside of the whole sequence
    # whose size is specified.
    xp = cuda.get_array_module(anchor)  # anchor shape = T * A, 2

    index_inside = xp.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] <= seq_len)
    )[0]
    return index_inside
