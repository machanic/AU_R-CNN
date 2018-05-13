'''
author : machen
'''
import numpy as np

from chainer import cuda
from AU_rcnn.utils.bbox.bbox_iou import bbox_intersection_area
from warnings import warn

class ProposalMultiLabel(object):
    """filter bad background (AU label = 0) target
       the only foreground bbox should use bbox which the labeled as meaningful AU

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.
    Args:
        n_sample (int): The number of sampled regions. which equals posive region + negative region
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_lo`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=12,
                 pos_ratio=0.5,
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, label):
        """
        Note: this legacy only handle bbox which is in only one image's bbox and label
        filter ground truth bbox which have 'bad' AU=0 label box
        the bad bbox means which contains or almost contains another bbox but label =0,
        this means the couple have large intersection area between them.

        This legacy samples total of :obj:`self.n_sample` RoIs
        from the :obj bbox.
        The RoIs are assigned with the ground truth class labels
        as many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.


        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is the number of foreground classes. it is equals to len('01010111..') so label all AU=0,
            which means no action move
        Args:
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',L)`. Its range is :math:`[-1, 0, 1]`
        Returns:
            (array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.

            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        xp = cuda.get_array_module(bbox)
        if xp != np:
            bbox = cuda.to_cpu(bbox)
            label = cuda.to_cpu(label)

        n_bbox, _ = bbox.shape
        intersection_area = bbox_intersection_area(bbox, bbox)
        xp.fill_diagonal(intersection_area, 0)
        cal_area = lambda y_min, x_min, y_max, x_max: (y_max - y_min) * (x_max-x_min)
        bad_box_index_set = set()
        # sorted then unique
        interarea_bbox_i_j = set(map(tuple, map(sorted,zip(*xp.nonzero(intersection_area)))))
        all_zero_func = lambda array: len(np.where(array > 0)[0]) == 0
        # filter which label = 0 but contains another box, only non-zero will compare each other

        for i, j in interarea_bbox_i_j:
            area = intersection_area[i, j]
            small_box_inter_area_prop = np.max((area/ float(cal_area(*bbox[i])), area/ float(cal_area(*bbox[j]))))
            if small_box_inter_area_prop < 0.8: # big than 0.8 overlap will choose to filter bad label bbox
                continue
            small_box_idx = np.argmax((area/float(cal_area(*bbox[i])), area/float(cal_area(*bbox[j]))))
            big_box_idx = 1 - small_box_idx
            label_small_box = label[np.array([i, j])[small_box_idx]]
            label_small_all_zeros = all_zero_func(label_small_box)
            label_big_box = label[np.array([i, j])[big_box_idx]]
            label_big_all_zeros = all_zero_func(label_big_box)
            # 删掉的应该是只带有-1未知的和只带有0
            if not label_small_all_zeros and label_big_all_zeros:  # only big box with background label will drop
                big_box_index = np.array([i, j])[big_box_idx]
                bad_box_index_set.add(big_box_index)

        filtered_bbox = []
        filtered_label = []
        for box_idx, box in enumerate(bbox):
            if box_idx not in bad_box_index_set:
                filtered_bbox.append(box)
                filtered_label.append(label[box_idx])

        bbox = np.array(filtered_bbox)
        label = np.array(filtered_label)

        if xp != np:
            bbox = cuda.to_gpu(bbox)
            label = cuda.to_gpu(label)
        return bbox, label
