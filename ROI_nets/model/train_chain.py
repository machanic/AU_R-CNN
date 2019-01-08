import numpy as np

import chainer
import chainer.functions as F

from chainer import cuda
import config

class TrainChain(chainer.Chain):

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

    def __init__(self, backbone):
        super(TrainChain, self).__init__()

        with self.init_scope():
            self.backbone = backbone
            self.neg_pos_ratio = 3


    def __call__(self, imgs, label):
        """Forward backbone and sigmoid cross entropy loss.
        support batch_size > 1 train
        Here are notations used.
        * :math:`N` is the batch size.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images. shape is (N C H W)


        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        xp = cuda.get_array_module(imgs)
        with cuda.get_device_from_array(imgs) as device:

            _, _, H, W = imgs.shape
            score = self.backbone(imgs)

            # 算sigmoid的时候，从gt中挑选=1的元素，然后从pred=1 但gt=0中挑选元素，如果还不够再从剩下的随机挑选凑够x 3倍的=0元素，最后算sigmoid cross entropy
            union_gt = set()  # union of prediction positive and ground truth positive
            cpu_gt_roi_label = chainer.cuda.to_cpu(label)
            gt_pos_index = np.nonzero(cpu_gt_roi_label)
            cpu_pred_score = (chainer.cuda.to_cpu(score.data) > 0).astype(np.int32)
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
                rest_pick_count = min(rest_pick_count, len(gt_neg_index_array))
                choice_rest = np.random.choice(np.arange(len(gt_neg_index_array)), size=rest_pick_count, replace=False)
                gt_pos_index_lst.extend(list(map(tuple,gt_neg_index_array[choice_rest].tolist())))

            pick_index = list(zip(*gt_pos_index_lst))
            if len(union_gt) == 0:
                accuracy_pick_index = np.where(cpu_gt_roi_label)
            else:
                accuracy_pick_index = list(zip(*union_gt))
            accuracy = F.binary_accuracy(score[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         label[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])
            loss = F.sigmoid_cross_entropy(score,
                                           label)

            chainer.reporter.report({
                'loss': loss, "accuracy": accuracy},
                self)
            return loss





