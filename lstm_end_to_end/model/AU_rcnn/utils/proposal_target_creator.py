import numpy as np

from chainer import cuda
from AU_rcnn.utils.bbox.bbox_iou import bbox_intersection_area
from warnings import warn
class ProposalTargetCreator(object):
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
                 n_sample=8,
                 pos_ratio=0.25,
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio

    # did not used yet
    def filter_overlap(self, box_array):
        '''
        filter by overlap area
        :param box_array: the box_array to filter
        :return: box_array which only contain box which is smaller between 2 box overlap (keep box which
        intersection area is bigger propotional compare to itself area)
        '''
        warn("deprecated method. not used yet")
        # each box is (y_min, x_min, y_max, x_max)
        tl = np.maximum(box_array[:, None, :2], box_array[:, :2])
        br = np.minimum(box_array[:, None, 2:], box_array[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(
            axis=2)  # area_i is n x n matrix, each is box_i vs box_j intersect area
        area_i = np.squeeze(area_i)
        index_max_array = area_i.argmax(axis=1)
        inter_area_b = area_i.max(axis=1)
        filter_list = set()
        area = lambda y_min, x_min, y_max, x_max: (y_max - y_min) * (x_max - x_min)
        for index, box in enumerate(box_array):
            propotional = area(box) / inter_area_b[index]
            propotional_b = area(box_array[index_max_array[index]]) / inter_area_b[index]
            if propotional > 0.7:
                filter_list.add(box)
            elif propotional_b > 0.7:
                filter_list.add(box_array[index_max_array[index]])
            else:  # only little intesection area
                filter_list.add(box)
                filter_list.add(box_array[index_max_array[index]])

        return filter_list

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
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L]`, where
                :math:`L` is the number of foreground classes. so label contains AU=0, which means no action move


        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.

            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        xp = cuda.get_array_module(bbox)

        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)

        n_bbox, _ = bbox.shape
        intersection_area = bbox_intersection_area(bbox, bbox)
        xp.fill_diagonal(intersection_area, 0)
        cal_area = lambda y_min, x_min, y_max, x_max: (y_max - y_min) * (x_max-x_min)
        bad_box_index_set = set()
        # sorted then unique
        interarea_bbox_i_j = set(map(tuple, map(sorted,zip(*xp.nonzero(intersection_area)))))

        # filter which label = 0 but contains another box, only non-zero will compare each other

        for i, j in interarea_bbox_i_j:
            area = intersection_area[i, j]
            small_box_inter_area_prop = np.max((area/ float(cal_area(*bbox[i])), area/ float(cal_area(*bbox[j]))))
            if small_box_inter_area_prop < 0.9: # big than 0.8 overlap will choose to filter bad label bbox
                continue
            small_box_idx = np.argmax((area/float(cal_area(*bbox[i])), area/float(cal_area(*bbox[j]))))
            big_box_idx = 1 - small_box_idx
            label_small_box = label[np.array([i, j])[small_box_idx]]
            label_big_box = label[np.array([i, j])[big_box_idx]]
            if label_big_box == 0 and label_small_box != 0:  # only big box with background label will drop
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

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)  # 按照一定比例生成正例
        # Select foreground RoIs as those with >= pos_iou_thresh IoU. IoU刷掉了一批不合适的ROI，从roi_bbox混合列表去选
        pos_index = np.where(label > 0)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi). 比较小的一定阈值之内的IoU视为负的label
        neg_index = np.where(label == 0)[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # 因为是concat起来，所以正例数量一过，可以强行设置为负的label。negative labels --> 0
        sample_roi = bbox[keep_index]

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_label = cuda.to_gpu(gt_roi_label)
        return sample_roi, gt_roi_label
