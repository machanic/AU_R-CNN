import chainer
import numpy as np
from two_stream_rgb_flow.constants.enum_type import TwoStreamMode
import chainer.functions as F
import chainer.links as L

class Wrapper(chainer.Chain):

    def __init__(self, au_rcnn_train_chain_list, n_class, database, T, two_stream_mode, gpus):
        self.database = database
        self.T = T
        self.two_stream_mode = two_stream_mode
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

                self.conv_reduce = L.Convolution2D(2048 * 2, 2048, 1, 1, pad=0, nobias=True).to_gpu(gpus[0])

            self.fc = L.Linear(2048, 1024).to_gpu(gpus[0])
            self.score = L.Linear(1024, n_class).to_gpu(gpus[0])


    def extract_batch(self, rgb_images, flow_images, bboxes):  # can only support optical flow extract
        # rgb_images = (B,C,H,W) flow_images = (B,T,C,H,W), bboxes = (B,F,4)
        self.flow_gpu = self.rgb_gpu
        self.au_rcnn_train_chain_flow = self.au_rcnn_train_chain_flow.to_gpu(self.rgb_gpu)
        rgb_feature, flow_feature, roi_feature = self.get_roi_feature(rgb_images, flow_images, bboxes, extract_rgb_flow=True)  # B, F, 2048, 7, 7
        batch_size, frame_box, *_ = roi_feature.shape

        rgb_feature = rgb_feature.reshape(batch_size * frame_box, 2048, 7, 7)
        rgb_feature = F.average_pooling_2d(rgb_feature, ksize=7, stride=1)
        rgb_feature = rgb_feature.reshape(batch_size, frame_box, 2048)

        flow_feature = flow_feature.reshape(batch_size * frame_box, 2048, 7, 7)
        flow_feature = F.average_pooling_2d(flow_feature, ksize=7, stride=1)
        flow_feature = flow_feature.reshape(batch_size, frame_box, 2048)

        roi_feature = roi_feature.reshape(batch_size * frame_box, 2048, 7, 7)
        roi_feature = F.average_pooling_2d(roi_feature, ksize=7, stride=1)
        roi_feature = roi_feature.reshape(batch_size, frame_box, 2048)

        return rgb_feature, flow_feature, roi_feature


    def predict(self, roi_features):  # B, T, F, 12
        with chainer.cuda.get_device_from_array(roi_features.data) as device:
            pred = chainer.cuda.to_cpu(roi_features.data)  # B, T, F, class_num
            pred = (pred > 0).astype(np.int32)
            return pred

    def get_loss(self, roi_feature, gt_roi_label):
        neg_pos_ratio = 3
        with chainer.cuda.get_device_from_array(roi_feature.data) as device:
            batch, frame_box, channel, roi_height, roi_width = roi_feature.shape
            roi_feature = F.reshape(roi_feature, shape=(batch * frame_box, channel, roi_height, roi_width))
            roi_feature = F.average_pooling_2d(roi_feature, ksize=7, stride=1)
            roi_feature = roi_feature.reshape(batch * frame_box, 2048)  # #  B * F, 2048, 7, 7

            predict_score = F.relu(self.fc(roi_feature))
            predict_score = self.score(predict_score)

            gt_roi_label = gt_roi_label.reshape(-1, gt_roi_label.shape[-1])
            assert predict_score.shape == gt_roi_label.shape, \
                "{0} != {1} (pred!=gt)".format(predict_score.shape, gt_roi_label.shape)
            union_gt = set()  # union of prediction positive and ground truth positive
            cpu_gt_roi_label = chainer.cuda.to_cpu(gt_roi_label)
            gt_pos_index = np.nonzero(cpu_gt_roi_label)
            cpu_pred_score = (chainer.cuda.to_cpu(roi_feature.data) > 0).astype(np.int32)
            pred_pos_index = np.nonzero(cpu_pred_score)
            len_gt_pos = len(gt_pos_index[0]) if len(gt_pos_index[0]) > 0 else 1
            neg_pick_count = neg_pos_ratio * len_gt_pos
            gt_pos_index_set = set(list(zip(*gt_pos_index)))
            pred_pos_index_set = set(list(zip(*pred_pos_index)))
            union_gt.update(gt_pos_index_set)
            union_gt.update(pred_pos_index_set)
            false_positive_index = np.asarray(list(pred_pos_index_set - gt_pos_index_set))  # shape = n x 2
            gt_pos_index_lst = list(gt_pos_index_set)
            if neg_pick_count <= len(false_positive_index):
                choice_fp = np.random.choice(np.arange(len(false_positive_index)), size=neg_pick_count, replace=False)
                gt_pos_index_lst.extend(list(map(tuple, false_positive_index[choice_fp].tolist())))
            else:
                gt_pos_index_lst.extend(list(map(tuple, false_positive_index.tolist())))
                rest_pick_count = neg_pick_count - len(false_positive_index)
                gt_neg_index = np.where(cpu_gt_roi_label == 0)
                gt_neg_index_set = set(list(zip(*gt_neg_index)))
                gt_neg_index_set = gt_neg_index_set - set(gt_pos_index_lst)  # remove already picked
                gt_neg_index_array = np.asarray(list(gt_neg_index_set))
                rest_pick_count = len(gt_neg_index_array) if len(gt_neg_index_array) < rest_pick_count else rest_pick_count
                choice_rest = np.random.choice(np.arange(len(gt_neg_index_array)), size=rest_pick_count, replace=False)
                gt_pos_index_lst.extend(list(map(tuple, gt_neg_index_array[choice_rest].tolist())))
            pick_index = list(zip(*gt_pos_index_lst))
            if len(union_gt) == 0:
                accuracy_pick_index = np.where(cpu_gt_roi_label)
            else:
                accuracy_pick_index = list(zip(*union_gt))
            accuracy = F.binary_accuracy(predict_score[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                         gt_roi_label[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])
            loss = F.sigmoid_cross_entropy(predict_score[list(pick_index[0]), list(pick_index[1])],
                                           gt_roi_label[list(pick_index[0]), list(pick_index[1])])  # 支持多label

            chainer.reporter.report({
                'loss': loss, "accuracy": accuracy},
                self)
        return loss, accuracy

    def get_roi_feature(self, rgb_images, flow_images, bboxes, extract_rgb_flow=False):
        # rgb_images = B, C, H, W;
        # flow_images = B, T, C, H, W;
        # boxes = B, F, 4; where B = T x original_batch_size. F is box number in each frame
        # labels =  B, F, 12; where B = T x original_batch_size
        frame_box = bboxes.shape[1]

        if self.two_stream_mode == TwoStreamMode.rgb:
            batch, channel, height, width = rgb_images.shape
            with chainer.cuda.get_device_from_array(rgb_images.data) as device:
                roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_images, bboxes) # R, C, H, W
            roi_feature_rgb = roi_feature_rgb.reshape(batch, frame_box, 2048, 7, 7)  # B, F, 2048, 7, 7
            return roi_feature_rgb

        elif self.two_stream_mode == TwoStreamMode.optical_flow:
            # optical flow will only use x and y information
            batch, T, channel, height, width = flow_images.shape
            flow_images = flow_images.reshape(batch, T * 2, height, width)  # B, T*2, H, W

            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            bboxes = F.copy(bboxes, dst=self.flow_gpu)  # B, F, 4
            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, bboxes)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)  # R, 2048, 7, 7
            roi_feature_flow = roi_feature_flow.reshape(batch, frame_box, 2048, 7, 7)  # B, F, 2048, 7, 7
            return roi_feature_flow

        elif self.two_stream_mode == TwoStreamMode.rgb_flow:
            batch, T, channel, height, width = flow_images.shape
            flow_images = flow_images.reshape(batch, T * 2, height, width)  # B, T*2, H, W
            flow_images = F.copy(flow_images, dst=self.flow_gpu)
            flow_box = F.copy(bboxes, dst=self.flow_gpu)
            rgb_images = F.copy(rgb_images, dst=self.rgb_gpu)
            rgb_box = F.copy(bboxes, dst=self.rgb_gpu)

            with chainer.cuda.get_device_from_array(flow_images.data) as device:
                roi_feature_flow = self.au_rcnn_train_chain_flow(flow_images, flow_box)
            roi_feature_flow = F.copy(roi_feature_flow, dst=self.rgb_gpu)

            with chainer.cuda.get_device_from_array(rgb_images.data) as device:
                roi_feature_rgb = self.au_rcnn_train_chain_rgb(rgb_images, rgb_box)


            roi_feature = F.concat([roi_feature_rgb, roi_feature_flow], axis=1)  # R', 2*C, H, W
            with chainer.cuda.get_device_from_array(roi_feature.data) as device:
                roi_feature = self.conv_reduce(roi_feature)  # R', C, H ,W
            roi_feature = roi_feature.reshape(batch, frame_box, 2048, 7, 7)  # B, F, 2048, 7, 7

            if extract_rgb_flow:
                roi_feature_rgb_extract = roi_feature_rgb.reshape(batch, frame_box, 2048, 7, 7)
                roi_feature_flow_extract = roi_feature_flow.reshape(batch, frame_box, 2048, 7, 7)
                return roi_feature_rgb_extract, roi_feature_flow_extract, roi_feature

            return roi_feature

    def __call__(self, rgb_images, flow_images, bboxes, labels):
        roi_feature = self.get_roi_feature(
            rgb_images, flow_images, bboxes) #  B, F, 2048, 7, 7
        with chainer.cuda.get_device_from_array(roi_feature.data) as device:
            loss, accuracy = self.get_loss(roi_feature, labels)
        report_dict = {'loss': loss, "accuracy": accuracy}
        chainer.reporter.report(report_dict,
                                self)
        return loss
