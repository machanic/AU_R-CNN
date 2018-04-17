import chainer
import numpy as np

import config
from space_time_AU_rcnn.model.dynamic_AU_rcnn.dynamic_au_rcnn_train_chain import DynamicAU_RCNN_ROI_Extractor


class Wrapper(chainer.Chain):
    def __init__(self, au_rcnn_train_chain, loss_head_module, database, T, use_feature_map):
        self.database = database
        self.T = T
        self.use_feature_map = use_feature_map
        super(Wrapper, self).__init__()
        with self.init_scope():
            self.au_rcnn_train_chain = au_rcnn_train_chain
            self.loss_head_module = loss_head_module


    def get_roi_feature(self, images, bboxes, labels):
        if images.ndim == 4:
            assert images.shape[0] % self.T == 0
            # images shape = B, C, H, W; where B = T x original_batch_size
            # bboxes shape = B, F, 4; where B = T x original_batch_size. F is box number in each frame
            # labels shape = B, F, 12; where B = T x original_batch_size
            images = images.reshape(images.shape[0] // self.T, self.T, images.shape[1], images.shape[2], images.shape[3])
            bboxes = bboxes.reshape(bboxes.shape[0] // self.T, self.T, bboxes.shape[1], bboxes.shape[2])
            labels = labels.reshape(labels.shape[0] // self.T, self.T, labels.shape[1], labels.shape[2])
        # images shape = B, T, C, H, W
        # bboxes shape = B, T, F(9 or 8), 4
        # labels shape = B, T, F(9 or 8), 12
        batch, T, channel, height, width = images.shape
        if not isinstance(self.au_rcnn_train_chain, DynamicAU_RCNN_ROI_Extractor):
            images = images.reshape(batch * T, channel, height, width)  # B*T, C, H, W
            bboxes = bboxes.reshape(batch * T, config.BOX_NUM[self.database], 4)  # B*T, 9, 4
        roi_feature = self.au_rcnn_train_chain(images, bboxes)  # shape = B*T, F, D or B*T, F, C, H, W
        if not self.use_feature_map:
            roi_feature = roi_feature.reshape(batch, T, config.BOX_NUM[self.database], -1)  # shape = B, T, F, D
        else:
            # B*T*F, C, H, W => B, T, F, C, H, W
            roi_feature = roi_feature.reshape(batch, T, config.BOX_NUM[self.database], roi_feature.shape[-3],
                                              roi_feature.shape[-2], roi_feature.shape[-1])
        return roi_feature, labels

    def __call__(self, images, bboxes, labels):
        roi_feature, labels = self.get_roi_feature(images, bboxes, labels)
        loss, accuracy = self.loss_head_module(roi_feature, labels)
        report_dict = {'loss': loss, "accuracy": accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss

    # # can only predict one frame based on previous T-1 frame feature
    # def predict(self, images, bboxes):  # all shape is (B, T, F, D), but will only predict last frame output
    #     if not isinstance(images, chainer.Variable):
    #         images = chainer.Variable(images)
    #     with chainer.no_backprop_mode(), chainer.using_config('train', False):
    #         batch, T, channel, height, width = images.shape
    #         images = images.reshape(batch * T, channel, height, width)  # B*T, C, H, W
    #         bboxes = bboxes.reshape(batch * T, config.BOX_NUM[self.database], 4)  # B*T, 9, 4
    #         roi_feature = self.au_rcnn_train_chain.__call__(images, bboxes)  # shape = B*T, F, D
    #         roi_feature = roi_feature.reshape(batch, T, config.BOX_NUM[self.database], -1)  # shape = B, T, F, D
    #
    #         node_out = self.loss_head_module.forward(roi_feature)  # node_out B,T,F,D
    #         node_out = chainer.cuda.to_cpu(node_out.data)
    #         node_out = node_out[:, -1, :, :]  # B, F, D
    #         pred = (node_out > 0).astype(np.int32)
    #         pred = np.bitwise_or.reduce(pred, axis=1)  # B, D
    #
    #     return pred  # return batch x out_size, it is last time_step frame of 2-nd axis of input xs prediction
