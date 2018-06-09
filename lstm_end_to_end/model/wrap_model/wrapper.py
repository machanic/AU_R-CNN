import chainer
import numpy as np

import config
from lstm_end_to_end.constants.enum_type import TwoStreamMode
import chainer.functions as F

class Wrapper(chainer.Chain):

    def __init__(self, au_rcnn_train_chain, loss_head_module, database, T, use_feature_map, gpu):
        self.database = database
        self.T = T
        self.use_feature_map = use_feature_map
        self.gpu = gpu
        super(Wrapper, self).__init__()
        with self.init_scope():
            self.au_rcnn_train_chain = au_rcnn_train_chain
            self.loss_head_module = loss_head_module

    def reshape_roi_feature(self, mini_batch, T, roi_feature):
        if not self.use_feature_map:
            # shape = B, T, F, D
            roi_feature = roi_feature.reshape(mini_batch, T, config.BOX_NUM[self.database], -1)
        else:
            # B*T*F, C, H, W => B, T, F, C, H, W
            roi_feature = roi_feature.reshape(mini_batch, T, config.BOX_NUM[self.database], roi_feature.shape[-3],
                                              roi_feature.shape[-2], roi_feature.shape[-1])
        return roi_feature



    def get_roi_feature(self, rgb_images, bboxes, labels):
        if rgb_images.ndim == 4:
            assert rgb_images.shape[0] % self.T == 0
            # images shape = B, C, H, W; where B = T x original_batch_size
            # bboxes shape = B, F, 4; where B = T x original_batch_size. F is box number in each frame
            # labels shape = B, F, 12; where B = T x original_batch_size
            rgb_images = rgb_images.reshape(rgb_images.shape[0] // self.T, self.T, rgb_images.shape[1], rgb_images.shape[2], rgb_images.shape[3])
            bboxes = bboxes.reshape(bboxes.shape[0] // self.T, self.T, bboxes.shape[1], bboxes.shape[2])
            labels = labels.reshape(labels.shape[0] // self.T, self.T, labels.shape[1], labels.shape[2])
        # images shape = B, T, C, H, W
        # bboxes shape = B, T, F(9 or 8), 4
        # labels shape = B, T, F(9 or 8), 12
        batch, T, channel, height, width = rgb_images.shape

        rgb_images = rgb_images.reshape(batch * T, channel, height, width)  # B*T, C, H, W
        bboxes = bboxes.reshape(batch * T, config.BOX_NUM[self.database], 4)  # B*T, 9, 4
        with chainer.cuda.get_device_from_array(rgb_images.data) as device:
            roi_feature_rgb = self.au_rcnn_train_chain(rgb_images, bboxes)
        return self.reshape_roi_feature(batch, self.T, roi_feature_rgb), labels




    def __call__(self, images, flow_images, bboxes, labels):
        # roi_feature = B, T, F, D;  labels = B, T, F(9 or 8), 12
        roi_feature, labels = self.get_roi_feature(
            images, flow_images, bboxes, labels ) #  only optical flow output B, 1, F, D;  B, 1, F, 12
        loss, accuracy = self.loss_head_module(roi_feature, labels)
        report_dict = {'loss': loss, "accuracy": accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss
