from chainer import cuda

def bbox_intersection_area(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    xp = cuda.get_array_module(bbox_a)

    # top left , 因为两个bbox_a和bbox_b的shape不一样，box个数不一样，所以再第一个bbox_a插入一个维度，然后让bbox_b broadcast 到bbox_a
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # bbox_a[:, None, :2] insert [] outside each 4-tuple coordinate and slice start~ 2.
    # t1 bbox_a和bbox_b的 y_min , x_min里左上角最小的里面取得其中最大的坐标
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # 右下角最大的坐标里面取得其中最小的坐标

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2) # area_i

    return area_i  # n x k matrix, n is number of bbox_a, k is number of bbox_b

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This legacy accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
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
    # each bbox shape = (y_min,x_min,y_max,x_max)
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    xp = cuda.get_array_module(bbox_a)

    # top left , 因为两个bbox_a和bbox_b的shape不一样，box个数不一样，所以再第一个bbox_a插入一个维度，然后让bbox_b broadcast 到bbox_a
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # bbox_a[:, None, :2] insert [] outside each 4-tuple coordinate and slice start~ 2.
    # t1 bbox_a和bbox_b的 y_min , x_min里左上角最小的里面取得其中最大的坐标
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # 右下角最大的坐标里面取得其中最小的坐标

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2) # area_i
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


if __name__ == "__main__":
    import numpy as np
    from chainer import cuda
    bbox_a = np.array([(24.609375, 349.21875, 176.953125, 546.09375), (24.609375, 452.34375, 159.375, 546.09375)], dtype=np.float32)
    bbox_b = np.array([(24.609375, 452.34375, 159.375, 546.09375), (24.609375, 349.21875, 176.953125, 546.09375)] , dtype=np.float32)
    inter_area = bbox_intersection_area(bbox_a, bbox_b)
    np.fill_diagonal(inter_area, 0)
    inter_area = cuda.to_gpu(inter_area, cuda.get_device_from_id(0))
    interarea_bbox_i_j = set(map(tuple, map(sorted, zip(*np.nonzero(cuda.to_cpu(inter_area))))))
    for i,j in interarea_bbox_i_j:
        print(i,j)