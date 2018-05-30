from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import config
import six

def _mkanchors(ws, x_ctr):
    """
    Given a vector of widths (ws) around a center
    (x_ctr), output a set of anchors (windows). Note that x_ctr is scalar
    """

    ws = ws[:, np.newaxis]  # shape = n, 1
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       x_ctr + 0.5 * (ws - 1)))
    return anchors

def _whctrs(anchor):
    """
    # anchor is (x_min, x_max)
    Return width, x center for an anchor (window).
    """

    w = anchor[1] - anchor[0] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    return w, x_ctr

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, x_ctr = _whctrs(anchor) # x_min, x_max => w, x_ctr; default: 1, 0
    ws = w * scales  # scales = (8,16,32) or (config.ANCHOR_SCALE = (1,2,3,4,5,6,8,11,16))
    anchors = _mkanchors(ws, x_ctr)
    return anchors


def generate_anchor_base(base_size=16, anchor_scales=config.ANCHOR_SIZE):
    px = base_size / 2.
    anchor_base = np.zeros((len(anchor_scales), 2),
                           dtype=np.float32)
    for i in range(len(anchor_scales)):
        w = base_size * anchor_scales[i]
        anchor_base[i, 0] = px - w / 2.
        anchor_base[i, 1] = px + w / 2.
    return anchor_base

def get_all_anchors(time_seq_len, stride=config.ANCHOR_STRIDE, sizes=config.ANCHOR_SIZE):
    """
        Get all anchors in the largest possible image, shifted, floatbox
        sizes: 就是stride要扩大的倍数而已
        Returns:
            anchors: S x S x NUM_ANCHOR x 4, where S == MAX_SIZE, floatbox
            The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SCALE.

        """
    base_anchors = generate_anchor_base(stride, sizes) # shape=A,2  scale是anchor的scale，要跟stride去乘
    # 由向量变成矩阵，下面两句话难理解
    # 这里K = field_size x field_size = 83 * 83 = 6889
    shifts_x = np.arange(0, time_seq_len*stride, stride)  # [0, 16, 32, 48, 64, 80, ..., 1312]，为了将卷积后缩小16倍的图映射回原图的位置
    # Overview of Enumerate all shifted anchors:
    # add A anchors (1, A, 2) (where A = scale_num) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2), in the add we automatically used the numpy broadcast

    shifts = np.stack((shifts_x, shifts_x), axis=1)  # K,2,  K = field_size = seq_len
    K = shifts.shape[0]  # K = time_seq_len
    A = base_anchors.shape[0]
    field_of_anchors = (
            base_anchors.reshape((1, A, 2)) +
            shifts.reshape((1, K, 2)).transpose((1, 0, 2))) # return (K, A, 2)
    return field_of_anchors.astype(np.float32)


if __name__ == "__main__":
    field_anchors= get_all_anchors(10,sizes=[2,7,3])
    print(field_anchors)