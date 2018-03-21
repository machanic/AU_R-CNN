import numpy as np

import chainer
import chainer.functions as F

from chainer import cuda


class DynamicAU_RCNN_ROI_Extractor(chainer.Chain):

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
        faster_rcnn (~AU_rcnn.links.model.AU_rcnn.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
        proposal_target_creator_params: An instantiation of
            :obj:`AU_rcnn.links.model.AU_rcnn.ProposalTargetCreator`.

    """

    def __init__(self, au_rcnn):
        super(DynamicAU_RCNN_ROI_Extractor, self).__init__()
        with self.init_scope():
            self.au_rcnn = au_rcnn



    def __call__(self, imgs, bboxes):
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

        Returns:
            chainer.Variable roi feature:
            shape = N, R, 2048

        """
        xp = cuda.get_array_module(imgs)
        with cuda.get_device_from_array(imgs.data) as device:
            if isinstance(bboxes, chainer.Variable):
                bboxes = bboxes.data

            batch_size, seq_len, box_num, _ = bboxes.shape
            bboxes = bboxes.reshape(batch_size * seq_len, box_num, 4)
            _, _, _, H, W = imgs.shape

            features = self.au_rcnn.extractor(imgs)  # B, T, C, H, W
            # Since batch size is one, convert variables to singular form
            if batch_size * seq_len > 1:
                sample_roi_lst = []
                sample_roi_index_lst = []
                for n in range(batch_size * seq_len):
                    bbox = bboxes[n]  # bbox仍然是一个list，表示一个图内部的bbox
                    # 若其中的一个label为-99 表示是padding的值，此时该bbox是[-99,-99,-99,-99]
                    # bboxes[n] = bbox[bbox != xp.array(-99)].reshape(-1, 4)
                    # label = labels[n]
                    # labels[n] = label[label != xp.array(-99)].reshape(-1, class_number)
                    assert bbox.shape[0] > 0
                    # Sample RoIs and forward
                    sample_roi = bbox
                    sample_roi_index = n * np.ones((len(sample_roi),), dtype=xp.int32)
                    sample_roi_lst.extend(sample_roi)
                    sample_roi_index_lst.extend(sample_roi_index)
                sample_roi_lst = xp.stack(sample_roi_lst).astype(dtype=xp.float32)
                sample_roi_index_lst = xp.asarray(sample_roi_index_lst).astype(dtype=xp.int32)

            elif batch_size * seq_len == 1:  # batch_size = 1
                bbox = bboxes[0]
                sample_roi_lst = bbox
                sample_roi_index_lst = xp.zeros((len(bbox),), dtype=xp.int32)

            roi_feature = self.au_rcnn.head(
                features, sample_roi_lst, sample_roi_index_lst) # return B*T*F, C, H, W
            # Losses for outputs of the head.

        return roi_feature
