import chainer
import chainer.links as L

from time_axis_rcnn.model.time_segment_network.soi_pooling import soi_pooling_1d
import numpy as np
import chainer.functions as F


class FasterHeadModule(chainer.Chain):

    """
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
    """
    def __init__(self, in_channels, n_class, roi_size):
        # n_class includes the background
        super(FasterHeadModule, self).__init__()

        self.roi_size = roi_size
        with self.init_scope():
            self.fc6 = L.Linear(in_channels * roi_size, 1024)
            self.fc7 = L.Linear(1024, 1024)
            self.cls_loc = L.Linear(1024, n_class * 2)
            self.score = L.Linear(1024, n_class)

        self.n_class = n_class
        self.roi_size = roi_size

    def __call__(self, x, rois, roi_indices):
        """Forward the chain.
        We assume that there are :math:`N` batches.
        Args:
            x (~chainer.Variable): 3D: (n: batch, c: channel, w: width).
            rois (array): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 2)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (array): An array containing indices of featuremap_1d to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """

        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = soi_pooling_1d(
            x, indices_and_rois, self.roi_size, 1.0)  # shape = R', C, W, where R = number of ROI across batch and W = roi_size
        pool = pool.reshape(pool.shape[0], -1)  # R', C x roi_size
        fc6 = F.relu(self.fc6(pool))
        fc7 = F.relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

