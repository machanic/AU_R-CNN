import chainer
import numpy as np
import random
import config
from two_stream_rgb_flow.constants.enum_type import TwoStreamMode
import chainer.functions as F

class Wrapper(chainer.Chain):

    def __init__(self, au_rcnn_train_chain_list, loss_head_module, database, T, two_stream_mode, gpus, train_mode=True):
        self.database = database
        self.T = T
        self.two_stream_mode = two_stream_mode
        self.train_mode = train_mode
        super(Wrapper, self).__init__()
        with self.init_scope():
            if self.two_stream_mode == TwoStreamMode.rgb:
                self.au_rcnn_train_chain_rgb = au_rcnn_train_chain_list[0].to_gpu(gpus[0])
                self.rgb_gpu = gpus[0]
            elif self.two_stream_mode == TwoStreamMode.optical_flow:
                self.au_rcnn_train_chain_flow = au_rcnn_train_chain_list[0].to_gpu(gpus[0])
                self.flow_gpu = gpus[0]
            elif self.two_stream_mode == TwoStreamMode.rgb_flow:
                assert len(au_rcnn_train_chain_list) == 2
                self.au_rcnn_train_chain_rgb = au_rcnn_train_chain_list[0].to_gpu(gpus[0])
                self.rgb_gpu = gpus[0]
                self.au_rcnn_train_chain_flow = au_rcnn_train_chain_list[1].to_gpu(gpus[1])
                self.flow_gpu = gpus[1]
            if loss_head_module is not None:
                self.loss_head_module = loss_head_module.to_gpu(gpus[0])


    def extract_batch(self, rgb_images,  flow_images, bboxes, layer):  # can only support optical flow extract
        # rgb_images = (B,C,H,W) flow_images = (B,C,H,W), bboxes = (B,F,4)
        self.flow_gpu = self.rgb_gpu
        self.au_rcnn_train_chain_flow = self.au_rcnn_train_chain_flow.to_gpu(self.rgb_gpu)
        T = rgb_images.shape[0]
        assert T == bboxes.shape[0]
        rgb_images = rgb_images.reshape(1, T, rgb_images.shape[1], rgb_images.shape[2],
                                        rgb_images.shape[3])  # B(=1), T, C, H, W
        flow_images = flow_images.reshape(flow_images.shape[0] // self.T, self.T, flow_images.shape[1],
                                          flow_images.shape[2],
                                          flow_images.shape[3])  # B, T, C, H, W
        frame_box = bboxes.shape[1]
        assert frame_box == config.BOX_NUM[self.database]
        rgb_batch_size, T, channel, height, width = rgb_images.shape
        bboxes = bboxes.reshape(1, T, bboxes.shape[1], bboxes.shape[2]) # B, T, F, 4
        rgb_image = rgb_images.reshape(rgb_batch_size * T, channel, height, width)  # B*T, C, H, W
        rgb_box = bboxes.reshape(rgb_batch_size * T, frame_box, 4)  # B*T, F, 4
        with chainer.cuda.get_device_from_array(rgb_image.data) as device:
            roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_image, rgb_box)
        roi_feature_rgb = roi_feature_rgb.reshape(rgb_batch_size * T, frame_box, -1)

        flow_images = flow_images[:, :, :2, :, :]  # B, T, 2, H, W
        flow_images = flow_images.reshape(rgb_batch_size, self.T * 2, height, width)  # B, T*2, H, W
        flow_box = bboxes[:, -1, :, :]  # B, F, 4
        with chainer.cuda.get_device_from_array(flow_images.data) as device:
            roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, flow_box)
        roi_feature_flow = roi_feature_flow.reshape(rgb_batch_size, frame_box, -1)
        roi_feature_rgb  = chainer.cuda.to_cpu(roi_feature_rgb.data)
        roi_feature_flow = chainer.cuda.to_cpu(roi_feature_flow.data)
        return roi_feature_rgb, roi_feature_flow


    def get_roi_feature(self, rgb_images, flow_images, bboxes, labels):
        # images = B, C, H, W; where B = T x original_batch_size
        # boxes = B, F, 4; where B = T x original_batch_size. F is box number in each frame
        # labels =  B, F, 12; where B = T x original_batch_size
        assert rgb_images.ndim == flow_images.ndim == 4
        rgb_images = rgb_images.reshape(rgb_images.shape[0] // self.T, self.T, rgb_images.shape[1], rgb_images.shape[2],
                                        rgb_images.shape[3])    # B, T, C, H, W
        flow_images = flow_images.reshape(flow_images.shape[0] // self.T, self.T, flow_images.shape[1], flow_images.shape[2],
                                              flow_images.shape[3])  # B, T, C, H, W
        frame_box = bboxes.shape[1]
        assert frame_box == config.BOX_NUM[self.database]
        bboxes = bboxes.reshape(bboxes.shape[0] // self.T, self.T, bboxes.shape[1], bboxes.shape[2])
        labels = labels.reshape(labels.shape[0] // self.T, self.T, labels.shape[1], labels.shape[2])
        # images shape = B, T, C, H, W
        # bboxes shape = B, T, F(9 or 8), 4
        # labels shape = B, T, F(9 or 8), 12

        batch, T, channel, height, width = rgb_images.shape

        if self.two_stream_mode == TwoStreamMode.rgb:

            rgb_image = rgb_images.reshape(batch * T, channel, height, width)  # B*T, C, H, W
            box = bboxes.reshape(batch * T, frame_box, 4)  # B*T, F, 4 ; where T = offset
            with chainer.cuda.get_device_from_array(rgb_image.data) as device:
                roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_image, box)
            roi_feature_rgb = roi_feature_rgb.reshape(batch, T, frame_box, -1) # B, T, F, 2048
            return roi_feature_rgb, labels  # B, T, F, 12

        elif self.two_stream_mode == TwoStreamMode.optical_flow:
            # optical flow will only use x and y information
            flow_images = flow_images[:, :, 0:2, :, :]  # only use two channel x and y of optical flow image
            flow_images = flow_images.reshape(batch, T * 2, height, width)  # B, T*2, H, W

            box = bboxes[:, -1, :, :]  # B, F(9 or 8), 4
            label = labels[:, -1, :, :]  # B, F(9 or 8), 12

            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            box = F.copy(box, dst=self.flow_gpu)
            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, box)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)  # R, 2048
            roi_feature_flow = roi_feature_flow.reshape(batch, 1, frame_box, -1)  # B, T(=1), F, 2048
            return roi_feature_flow, label.reshape(batch, 1, frame_box, -1)

        elif self.two_stream_mode == TwoStreamMode.rgb_flow:
            flow_images = flow_images[:, :, 0:2, :, :]  # only use two channel x and y of optical flow image
            flow_images = flow_images.reshape(batch, T * 2, height, width)  # B, T*2, H, W

            flow_images = F.copy(flow_images, dst=self.flow_gpu)

            offset = random.randint(0, T - 1)  # 0 ~ T-1
            if not self.train_mode:
                offset = T // 2
            rgb_image = rgb_images[:, offset, :, :, :]  # B, C, H, W
            box = bboxes[:, offset, :, :]  # B, F, 4 ; where T = offset
            label = labels[:, offset, :, :]  # B, F, 12

            flow_box = F.copy(box, dst=self.flow_gpu)
            rgb_image = F.copy(rgb_image, dst=self.rgb_gpu)
            rgb_box = F.copy(box, dst=self.rgb_gpu)

            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, flow_box)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)

            with chainer.cuda.get_device_from_array(rgb_image.data) as device:
                roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_image, rgb_box)
                roi_feature_flow = roi_feature_flow.reshape(batch, 1, frame_box, -1)
                roi_feature_rgb = roi_feature_rgb.reshape(batch, 1, frame_box, -1)  # B, T(=1), F, 2048

                roi_feature = F.average(F.stack([roi_feature_rgb, roi_feature_flow]),axis=0,
                                        weights=chainer.Variable(self.au_rcnn_train_chain_rgb.xp.array([0.3, 0.7],
                                        dtype=roi_feature_flow.data.dtype)))
            return roi_feature, label.reshape(batch, 1, frame_box, -1)


    def __call__(self, rgb_images, flow_images, bboxes, labels):
        # roi_feature = B, T, F, D;  labels = B, T, F(9 or 8), 12
        roi_feature, labels = self.get_roi_feature(
            rgb_images, flow_images, bboxes, labels ) #  only optical flow output B, 1, F, D;  B, 1, F, 12
        with chainer.cuda.get_device_from_array(roi_feature.data) as device:
            loss, accuracy = self.loss_head_module(roi_feature, labels)
        report_dict = {'loss': loss, "accuracy": accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss
