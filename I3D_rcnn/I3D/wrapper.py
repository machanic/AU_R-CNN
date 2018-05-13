import chainer
import numpy as np

import config
from lstm_end_to_end.constants.enum_type import TwoStreamMode
import chainer.functions as F

class Wrapper(chainer.Chain):

    def __init__(self, roi_feature_extractor_list, loss_head_module, database, T, two_stream_mode, gpus):
        self.database = database
        self.T = T
        self.two_stream_mode = two_stream_mode
        super(Wrapper, self).__init__()
        with self.init_scope():
            if self.two_stream_mode == TwoStreamMode.rgb:
                self.roi_feature_extractor_rgb = roi_feature_extractor_list[0].to_gpu(gpus[0])
                self.rgb_gpu = gpus[0]
            elif self.two_stream_mode == TwoStreamMode.optical_flow:
                self.roi_feature_extractor_flow = roi_feature_extractor_list[0].to_gpu(gpus[0])
                self.flow_gpu = gpus[0]
            elif self.two_stream_mode == TwoStreamMode.spatial_temporal:
                assert len(roi_feature_extractor_list) == 2
                self.roi_feature_extractor_rgb = roi_feature_extractor_list[0].to_gpu(gpus[0])
                self.rgb_gpu = gpus[0]
                self.roi_feature_extractor_flow = roi_feature_extractor_list[1].to_gpu(gpus[1])
                self.flow_gpu = gpus[1]
            self.loss_head_module = loss_head_module.to_gpu(gpus[0])


    def get_roi_feature(self, rgb_images, flow_images, bboxes):

        # images shape = B, C, H, W; where B = T x original_batch_size
        # bboxes shape = B * T, F(9 or 8), 4
        if self.two_stream_mode == TwoStreamMode.rgb:
            rgb_images = rgb_images.reshape(rgb_images.shape[0] // self.T, self.T, rgb_images.shape[1],
                                            rgb_images.shape[2],
                                            rgb_images.shape[3])  # B, T, C, H, W
            rgb_images = F.transpose(rgb_images, axes=(0, 2, 1, 3, 4))  # B, C, T, H, W
            mini_batch, channel, seq_len, height, width = rgb_images.shape
            rgb_images = F.copy(rgb_images, dst=self.rgb_gpu)
            bboxes = F.copy(bboxes, dst=self.rgb_gpu)
            with chainer.cuda.get_device_from_array(rgb_images.data) as device:
                roi_feature_rgb = self.roi_feature_extractor_rgb(rgb_images, bboxes) #  R',2048, where R'=B*T*F
            return F.reshape(roi_feature_rgb, shape=(mini_batch, seq_len, config.BOX_NUM[self.database], -1)) # B,T,F,2048

        elif self.two_stream_mode == TwoStreamMode.optical_flow:
            # optical flow will only use x and y information
            flow_images = flow_images.reshape(flow_images.shape[0] // self.T, self.T, flow_images.shape[1],
                                              flow_images.shape[2], flow_images.shape[3])  # B, T, C, H, W
            flow_images = flow_images[:, :, :2, :, :]
            flow_images = F.transpose(flow_images, (0, 2, 1, 3, 4))  # B, C, T, H, W
            mini_batch, channel, seq_len, height, width = flow_images.shape  # B, C, T, H, W
            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            bboxes = F.copy(bboxes, dst=self.flow_gpu)
            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.roi_feature_extractor_flow(flow_images, bboxes)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)
            return F.reshape(roi_feature_flow, shape=(mini_batch, seq_len, config.BOX_NUM[self.database], -1)) # B,T,F,2048

        elif self.two_stream_mode == TwoStreamMode.spatial_temporal:
            flow_images = flow_images.reshape(flow_images.shape[0] // self.T, self.T, flow_images.shape[1],
                                              flow_images.shape[2], flow_images.shape[3])  # B, T, C, H, W
            flow_images = flow_images[:, :, :2, :, :]
            flow_images = F.transpose(flow_images, (0, 2, 1, 3, 4))  # B, C, T, H, W
            mini_batch, channel, seq_len, height, width = flow_images.shape  # B, C, T, H, W
            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            flow_bboxes = F.copy(bboxes, dst=self.flow_gpu)

            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.roi_feature_extractor_flow(flow_images, flow_bboxes)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)
            roi_feature_flow = F.reshape(roi_feature_flow, shape=(mini_batch, seq_len, config.BOX_NUM[self.database], -1))

            rgb_images = rgb_images.reshape(rgb_images.shape[0] // self.T, self.T, rgb_images.shape[1],
                                            rgb_images.shape[2],
                                            rgb_images.shape[3])  # B, T, C, H, W
            rgb_images = F.transpose(rgb_images, axes=(0, 2, 1, 3, 4))  # B, C, T, H, W
            mini_batch, channel, seq_len, height, width = rgb_images.shape
            rgb_images = F.copy(rgb_images, dst=self.rgb_gpu)
            bboxes = F.copy(bboxes, dst=self.rgb_gpu)
            with chainer.cuda.get_device_from_array(rgb_images.data) as device:
                roi_feature_rgb = self.roi_feature_extractor_rgb(rgb_images, bboxes)  # R',2048, where R'=B*T*F

            roi_feature_rgb = F.reshape(roi_feature_rgb, shape=(mini_batch, seq_len, config.BOX_NUM[self.database], -1))

            return F.average([roi_feature_rgb, roi_feature_flow], weights=self.xp.array([0.4,0.6]),keepdims=True)



    def __call__(self, images, flows, bboxes, labels):
        roi_feature, labels = self.get_roi_feature(images, flows, bboxes) # # B,T,F,2048
        loss, accuracy = self.loss_head_module(roi_feature, labels)
        report_dict = {'loss': loss, "accuracy": accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss
