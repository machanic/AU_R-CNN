from chainer import cuda

def decode_segment_target(src_seg, loc): # in paper fomular (2). from offset and anchor back to real coordinate
    """Decode bounding boxes from bounding box offsets and scales.

        Given bounding box offsets and scales computed by
        :meth:`bbox2loc`, this function decodes the representation to
        coordinates in 2D image coordinates.

        Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
        box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
        the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
        and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
        by the following formulas.
        下面公式的p代表src_seg, t_y等代表loc
        * :math:`\\hat{g}_y = p_h t_y + p_y`
        * :math:`\\hat{g}_x = p_w t_x + p_x`
        * :math:`\\hat{g}_h = p_h \\exp(t_h)`
        * :math:`\\hat{g}_w = p_w \\exp(t_w)`

        The decoding formulas are used in works such as R-CNN [#]_.

        The output is same type as the type of the inputs.

        .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
        Rich feature hierarchies for accurate object detection and semantic \
        segmentation. CVPR 2014.

        Args:
            src_seg (array): A coordinates of bounding boxes.
                Its shape is :math:`(R, 2)`. These coordinates are used to
                compute :math:`p_x,  p_w`.
            loc (array): An array with offsets and scales.
                The shapes of :obj:`src_seg` and :obj:`loc` should be same.
                This contains values :math:`t_x, t_w`.

        Returns:
            np.array:
            Decoded bounding box coordinates. Its shape is :math:`(R, 2)`. \
            The second axis contains four values \
            :math:`\\hat{g}_y, \\hat{g}_x, \\hat{g}_h, \\hat{g}_w`.

        """
    xp = cuda.get_array_module(src_seg)

    if src_seg.shape[0] == 0:
        return xp.zeros((0, 2), dtype=loc.dtype)

    src_seg = src_seg.astype(src_seg.dtype, copy=False)

    src_width = src_seg[:, 1] - src_seg[:, 0]  # shape = (R,)
    src_ctr_x = src_seg[:, 0] + 0.5 * src_width  # shape = (R,)

    dx = loc[:, 0] # shape = (R,)
    dw = loc[:, 1]

    ctr_x = dx * src_width + src_ctr_x # shape = (R,)
    w = xp.exp(dw) * src_width # shape = (R,)

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)  # shape = (R, 2)
    dst_bbox[:, 0] = ctr_x - 0.5 * w
    dst_bbox[:, 1] = ctr_x + 0.5 * w

    return dst_bbox  # shape = (R, 2)



def encode_segment_target(src_seg, dst_seg):
    # 将gt_segments和anchors的偏差转为中心坐标和width和height的编码值
    """Encodes the source and the destination bounding segments to "loc".

    Given bounding segments, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_seg (np.array): An image coordinate array whose shape is
            :math:`(R, 2)`. :math:`R` is the number of bounding boxes.
            These coordinates are used to compute :math:`p_y, p_x, p_h, p_w`.
        dst_seg (np.array): An image coordinate array whose shape is
            :math:`(R, 2)`.
            These coordinates are used to compute :math:`g_y, g_x, g_h, g_w`.

    Returns:
        np.array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 2)`.
        The second axis contains four values :math: `t_x, t_w`.

    """
    xp = cuda.get_array_module(src_seg)

    width = src_seg[:, 1] - src_seg[:, 0]  # shape = R
    ctr_x = src_seg[:, 0] + 0.5 * width  # shape = R

    base_width = dst_seg[:, 1] - dst_seg[:, 0]
    base_ctr_x = dst_seg[:, 0] + 0.5 * base_width

    dx = (base_ctr_x - ctr_x) / width  # shape =R
    dw = xp.log(base_width / width)  # shape =R

    loc = xp.vstack((dx, dw)).transpose()  # shape = R, 2; each row is (dx[i], dw[i])
    return loc


def segments_iou(seg_a, seg_b):
    #  seg_a (N,2); seg_b (K,2)
    #
    xp = cuda.get_array_module(seg_a)
    zero_a = xp.zeros((seg_a.shape[0], 1), dtype=seg_a.dtype)
    zero_b = xp.zeros((seg_b.shape[0], 1), dtype=seg_b.dtype)
    seg_a_xmin, seg_a_xmax = xp.split(seg_a, indices_or_sections=2, axis=1)
    bbox_a = xp.hstack((zero_a, seg_a_xmin, zero_a, seg_a_xmax),axis=1)  # N, 4

    seg_b_xmin, seg_b_xmax = xp.split(seg_b, indices_or_sections=2, axis=1)
    bbox_b = xp.hstack((zero_b, seg_b_xmin, zero_b, seg_b_xmax),axis=1) # K, 4

    return bbox_iou(bbox_a, bbox_b)


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    xp = cuda.get_array_module(bbox_a)

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)
