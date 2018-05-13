import chainer
import numpy as np

import config
from lstm_end_to_end.constants.enum_type import TwoStreamMode
import chainer.functions as F

class Wrapper(chainer.Chain):

    def __init__(self, au_rcnn_train_chain_list, loss_head_module, database, T, use_feature_map, two_stream_mode, gpus):
        self.database = database
        self.T = T
        self.use_feature_map = use_feature_map
        self.two_stream_mode = two_stream_mode
        super(Wrapper, self).__init__()
        with self.init_scope():
            if self.two_stream_mode == TwoStreamMode.rgb:
                self.au_rcnn_train_chain_rgb = au_rcnn_train_chain_list[0].to_gpu(gpus[0])
                self.rgb_gpu = gpus[0]
            elif self.two_stream_mode == TwoStreamMode.optical_flow:
                self.au_rcnn_train_chain_flow = au_rcnn_train_chain_list[0].to_gpu(gpus[0])
                self.flow_gpu = gpus[0]
            elif self.two_stream_mode == TwoStreamMode.spatial_temporal:
                assert len(au_rcnn_train_chain_list) == 2
                self.au_rcnn_train_chain_rgb = au_rcnn_train_chain_list[0].to_gpu(gpus[0])
                self.rgb_gpu = gpus[0]
                self.au_rcnn_train_chain_flow = au_rcnn_train_chain_list[1].to_gpu(gpus[1])
                self.flow_gpu = gpus[1]
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

    # FIXME 必须用I3D模型修改掉
    def extract_batch(self, images, bbox, layer):  # can only support optical flow extract
        xp = self.au_rcnn_train_chain_flow.xp
        T = images.shape[0]
        assert self.T == T
        images = images.reshape(1, images.shape[0], images.shape[1], images.shape[2], images.shape[3]) # 1, T, C, H, W
        flow_images = images[:, :, 0:2, :, :]  # only use two channel x and y of optical flow image
        flow_images = flow_images.reshape(1, T * 2, images.shape[-2], images.shape[-1])
        x = chainer.Variable(xp.asarray(flow_images))  # shape = (1,T * 2,H,W)
        bbox = chainer.Variable(xp.asarray([bbox]))  # shape = (1, box_num, 4)
        with chainer.cuda.get_device_from_array(x.data) as device:
            roi_feature_flow = self.au_rcnn_train_chain_flow(x, bbox)
        return roi_feature_flow

    def get_roi_feature(self, rgb_images, flow_images, bboxes, labels):
        if rgb_images.ndim == 4:
            assert rgb_images.shape[0] % self.T == 0
            # images shape = B, C, H, W; where B = T x original_batch_size
            # bboxes shape = B, F, 4; where B = T x original_batch_size. F is box number in each frame
            # labels shape = B, F, 12; where B = T x original_batch_size
            rgb_images = rgb_images.reshape(rgb_images.shape[0] // self.T, self.T, rgb_images.shape[1], rgb_images.shape[2], rgb_images.shape[3])
            flow_images = flow_images.reshape(flow_images.shape[0] // self.T, self.T, flow_images.shape[1], flow_images.shape[2],
                                              flow_images.shape[3])
            bboxes = bboxes.reshape(bboxes.shape[0] // self.T, self.T, bboxes.shape[1], bboxes.shape[2])
            labels = labels.reshape(labels.shape[0] // self.T, self.T, labels.shape[1], labels.shape[2])
        # images shape = B, T, C, H, W
        # bboxes shape = B, T, F(9 or 8), 4
        # labels shape = B, T, F(9 or 8), 12
        batch, T, channel, height, width = rgb_images.shape

        if self.two_stream_mode == TwoStreamMode.rgb:
            rgb_images = rgb_images.reshape(batch * T, channel, height, width)  # B*T, C, H, W
            bboxes = bboxes.reshape(batch * T, config.BOX_NUM[self.database], 4)  # B*T, 9, 4
            rgb_images = F.copy(rgb_images, dst=self.rgb_gpu)
            bboxes = F.copy(bboxes, dst=self.rgb_gpu)
            with chainer.cuda.get_device_from_array(rgb_images.data) as device:
                roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_images, bboxes)
            return self.reshape_roi_feature(batch, self.T, roi_feature_rgb), labels

        elif self.two_stream_mode == TwoStreamMode.optical_flow:
            # optical flow will only use x and y information
            flow_images = flow_images[:, :, 0:2, :, :]  # only use two channel x and y of optical flow image
            flow_images = flow_images.reshape(batch, T * 2, height, width)  # B, T*2, H, W
            bboxes = bboxes[:, -1, :, :]  # B, F(9 or 8), 4
            labels = labels[:, -1, :, :]  # B, F(9 or 8), 12

            T = 1  # only one bboxes and labels will be used
            labels = labels.reshape(batch, T, config.BOX_NUM[self.database], -1)

            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            bboxes = F.copy(bboxes, dst=self.flow_gpu)
            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, bboxes)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)
            return self.reshape_roi_feature(batch, T, roi_feature_flow), labels

        elif self.two_stream_mode == TwoStreamMode.spatial_temporal:
            flow_images = flow_images[:, :, 0:2, :, :]  # only use two channel x and y of optical flow image
            flow_images = flow_images.reshape(batch, T * 2, height, width)  # B, T*2, H, W
            bboxes = bboxes[:, -1, :, :]  # B, F(9 or 8), 4
            labels = labels[:, -1, :, :]  # B, F(9 or 8), 12

            T = 1
            labels = labels.reshape(batch, T, config.BOX_NUM[self.database], -1)

            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            flow_bboxes = F.copy(bboxes, dst=self.flow_gpu)
            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, flow_bboxes)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)

            bboxes = F.copy(bboxes, dst=self.rgb_gpu) # B, F(9 or 8), 4
            rgb_images = rgb_images[:, -1, :, :, :]  # B, C, H, W
            rgb_images = F.copy(rgb_images, dst=self.rgb_gpu)
            with chainer.cuda.get_device_from_array(rgb_images.data) as device:
                roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_images, bboxes)
            roi_feature_rgb = F.copy(roi_feature_rgb, dst=self.rgb_gpu)
            roi_feature = F.average([roi_feature_rgb, roi_feature_flow], weights=chainer.Variable(self.xp.array([0.4,0.6])))
            return self.reshape_roi_feature(batch, T, roi_feature), labels


    def __call__(self, images, flow_images, bboxes, labels):
        # roi_feature = B, T, F, D;  labels = B, T, F(9 or 8), 12
        roi_feature, labels = self.get_roi_feature(
            images,flow_images, bboxes, labels ) #  only optical flow output B, 1, F, D;  B, 1, F, 12
        loss, accuracy = self.loss_head_module(roi_feature, labels)
        report_dict = {'loss': loss, "accuracy": accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss
