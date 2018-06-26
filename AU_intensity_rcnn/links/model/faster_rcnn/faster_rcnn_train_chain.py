import numpy as np

import chainer
import chainer.functions as F

from chainer import cuda
import config
from scipy.stats.stats import pearsonr

class FasterRCNNTrainChain(chainer.Chain):

    """Calculate losses for Faster R-CNN and report them.

    This is used to train Faster R-CNN in the joint training scheme
    [#FRCNN]_.

    The losses include:

    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    .. [#FRCNN] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        faster_rcnn (~AU_rcnn.links.model.faster_rcnn.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
        proposal_target_creator_params: An instantiation of
            :obj:`AU_rcnn.links.model.faster_rcnn.ProposalTargetCreator`.

    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.faster_rcnn = faster_rcnn

    def __call__(self, imgs, bboxes, labels):
        """Forward Faster R-CNN and calculate losses.
        support batch_size > 1 train
        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.
        * :math:`L` is the number of Action Unit Set(in config.py define)
        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images. shape is (N C H W)
            bboxes (~chainer.Variable): A batch of ground truth bounding boxes.
                Its shape is :math:`(N, R, 4)`.

            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R, L)`. this is for the multi-label region,
                 The background is excluded from the definition, which means that the range of the value
                is :math:`[-1,0,1]`.0 means this AU index does not occur. -1 means ignore_label
                classes.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        xp = cuda.get_array_module(imgs)
        with cuda.get_device_from_array(imgs) as device:
            if isinstance(bboxes, chainer.Variable):
                bboxes = bboxes.data
            if isinstance(labels, chainer.Variable):
                labels = labels.data

            batch_size = bboxes.shape[0]

            _, _, H, W = imgs.shape

            features = self.faster_rcnn.extractor(imgs)
            # Since batch size is one, convert variables to singular form
            if batch_size > 1:
                sample_roi_lst = []
                sample_roi_index_lst = []
                gt_roi_label = []
                for n in range(batch_size):
                    bbox = bboxes[n]  # bbox仍然是一个list，表示一个图内部的bbox
                    label = labels[n]  # label仍然是一个list，表示一个图内部的label
                    # 若其中的一个label为-99 表示是padding的值，此时该bbox是[-99,-99,-99,-99]
                    bbox = bbox[bbox != xp.array(-99)].reshape(-1, 4)
                    label = label[label != xp.array(-99)].reshape(-1, len(config.AU_INTENSITY_DICT))
                    assert label.shape[0] == bbox.shape[0] and bbox.shape[0] > 0
                    # Sample RoIs and forward
                    sample_roi = bbox
                    sample_roi_index = n * np.ones((len(sample_roi),), dtype=xp.int32)
                    gt_roi_label.extend(label)  # list 可以extend ndarray
                    sample_roi_lst.extend(sample_roi)
                    sample_roi_index_lst.extend(sample_roi_index)
                sample_roi_lst = xp.stack(sample_roi_lst).astype(dtype=xp.float32)
                gt_roi_label = xp.stack(gt_roi_label).astype(dtype=xp.int32)
                sample_roi_index_lst = xp.asarray(sample_roi_index_lst).astype(dtype=xp.int32)

            elif batch_size == 1:  # batch_size = 1
                bbox = bboxes[0]
                label = labels[0]
                sample_roi_lst, gt_roi_label = bbox, label
                sample_roi_index_lst = xp.zeros((len(sample_roi_lst),), dtype=xp.int32)

            roi_score = self.faster_rcnn.head(
                features, sample_roi_lst, sample_roi_index_lst)
            gt_roi_label = gt_roi_label.astype(xp.float32)
            # Losses for outputs of the head.
            assert roi_score.shape[0] == gt_roi_label.shape[0]
            pearson_correlation, _ = pearsonr(np.reshape(chainer.cuda.to_cpu(roi_score.data),-1),
                                              np.reshape(chainer.cuda.to_cpu(gt_roi_label),-1))
            loss = F.mean_squared_error(roi_score, gt_roi_label)
            chainer.reporter.report({
                                     'loss': loss, "pearson_correlation" : pearson_correlation},
                                    self)
        return loss





