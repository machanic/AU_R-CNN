import chainer
from time_axis_rcnn.model.time_segment_network.faster_rcnn_predictor import TimeSegmentRCNNPredictor
import numpy as np


class WrapperPredictor(chainer.Chain):

    # predict every frame based on segment predict result
    def __init__(self, time_seg_rcnn_predictor:TimeSegmentRCNNPredictor, class_num:int):
        super(WrapperPredictor, self).__init__()
        self.class_num = class_num
        with self.init_scope():
            self.seg_predictor = time_seg_rcnn_predictor



    def predict(self, featuremap):   # B, C, W

        assert featuremap.ndim == 3
        # predict_labels = (B, W)
        predict_labels = np.zeros(shape=(featuremap.shape[0],  featuremap.shape[2], self.class_num), dtype=np.int32)  # shape = (B, W, class)
        segment_list, labels, scores = self.seg_predictor.predict(featuremap)  # list of (R,2) list of (R,) list of R
        for batch_idx, (segment, label) in enumerate(zip(segment_list, labels)):
            for seg_id, (start_idx, end_idx) in enumerate(segment):
                predict_labels[batch_idx, int(round(start_idx)): int(round(end_idx)) + 1, label[seg_id]] = 1

        return predict_labels  #  (B, W, class)
