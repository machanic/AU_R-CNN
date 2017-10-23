import numpy as np

import chainer
import chainer.functions as F

from chainer import cuda
import config
import random
from AU_rcnn.utils.bbox.bbox_iou import bbox_iou
from dataset_toolkit.AU_balance_toolkit import load_AU_count,AU_repeat_level

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

    def __init__(self, faster_rcnn, use_sigmoid_cross_entropy, database, AU_count_name):
        super(FasterRCNNTrainChain, self).__init__()
        self.neg_pos_ratio = 3
        self.use_sigmoid_cross_entropy = use_sigmoid_cross_entropy

        # AU_count_path = config.DATA_PATH[database] + AU_count_name
        # AU_count = load_AU_count(AU_count_path)
        # AU_level = AU_repeat_level(5, AU_count)
        # self.AU_squeeze_level = dict()
        # for AU, level in AU_level.items():
        #     if AU in config.AU_ROI:
        #         self.AU_squeeze_level[config.AU_SQUEEZE.inv[AU]] = level
        with self.init_scope():
            self.faster_rcnn = faster_rcnn

    # def cleargrads(self):
    #     self.cleargrads()
    #     self.faster_rcnn.cleargrads()
    #     self.faster_rcnn.head.cleargrads()

    def reset_state(self):
        self.faster_rcnn.head.reset_state()

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
                    label = label[label != xp.array(-99)].reshape(-1, len(config.AU_SQUEEZE))
                    assert label.shape[0] == bbox.shape[0] and bbox.shape[0] > 0
                    # Sample RoIs and forward
                    sample_roi = bbox
                    sample_roi_index = n * xp.ones((len(sample_roi),), dtype=xp.int32)
                    gt_roi_label.extend(label)  # list 可以extend ndarray
                    sample_roi_lst.extend(sample_roi)
                    sample_roi_index_lst.extend(sample_roi_index)
                sample_roi_lst = xp.stack(sample_roi_lst).astype(dtype=xp.float32)
                gt_roi_label = xp.stack(gt_roi_label).astype(dtype=xp.int32)
                sample_roi_index_lst = xp.stack(sample_roi_index_lst).astype(dtype=xp.int32)

            elif batch_size == 1:  # batch_size = 1
                bbox = bboxes[0]
                label = labels[0]
                sample_roi_lst, gt_roi_label = bbox, label
                sample_roi_index_lst = xp.zeros((len(sample_roi_lst),), dtype=xp.int32)

            roi_score = self.faster_rcnn.head(
                features, sample_roi_lst, sample_roi_index_lst)
            # Losses for outputs of the head.
            assert roi_score.shape[0] == gt_roi_label.shape[0]

            if self.use_sigmoid_cross_entropy:
                # 算sigmoid的时候，从gt中挑选=1的元素，然后从pred=1 但gt=0中挑选元素，如果还不够再从剩下的随机挑选凑够x 3倍的=0元素，最后算sigmoid cross entropy
                union_gt = set()  # union of prediction positive and ground truth positive
                cpu_gt_roi_label = chainer.cuda.to_cpu(gt_roi_label)
                gt_pos_index = np.nonzero(cpu_gt_roi_label)
                cpu_pred_score = (chainer.cuda.to_cpu(roi_score.data) > 0).astype(np.int32)
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
                    gt_pos_index_lst.extend(list(map(tuple,false_positive_index[choice_fp].tolist())))
                else:
                    gt_pos_index_lst.extend(list(map(tuple, false_positive_index.tolist())))
                    rest_pick_count = neg_pick_count - len(false_positive_index)
                    gt_neg_index = np.where(cpu_gt_roi_label == 0)
                    gt_neg_index_set = set(list(zip(*gt_neg_index)))
                    gt_neg_index_set = gt_neg_index_set - set(gt_pos_index_lst)  # remove already picked
                    gt_neg_index_array = np.asarray(list(gt_neg_index_set))
                    choice_rest = np.random.choice(np.arange(len(gt_neg_index_array)), size=rest_pick_count, replace=False)
                    gt_pos_index_lst.extend(list(map(tuple,gt_neg_index_array[choice_rest].tolist())))
                # TODO need class imbalance?

                pick_index = list(zip(*gt_pos_index_lst))
                if len(union_gt) == 0:
                    accuracy_pick_index = np.where(cpu_gt_roi_label)
                else:
                    accuracy_pick_index = list(zip(*union_gt))
                accuracy = F.binary_accuracy(roi_score[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                             gt_roi_label[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])
                loss = F.sigmoid_cross_entropy(roi_score[list(pick_index[0]), list(pick_index[1])],
                                                   gt_roi_label[list(pick_index[0]), list(pick_index[1])])  # 支持多label
            else:
                accuracy = F.accuracy(roi_score, gt_roi_label)
                loss = F.softmax_cross_entropy(roi_score, gt_roi_label)
            chainer.reporter.report({
                                     'loss': loss, "accuracy": accuracy},
                                    self)
        return loss





