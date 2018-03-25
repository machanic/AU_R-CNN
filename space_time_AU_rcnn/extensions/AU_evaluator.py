import copy
import os
from collections import defaultdict

import chainer
import numpy as np
from chainer import DictSummary
from chainer import Reporter
from chainer.training.extensions import Evaluator
from overrides import overrides
from sklearn.metrics import f1_score

import config
from space_time_AU_rcnn.model.dynamic_AU_rcnn.dynamic_au_rcnn_train_chain import DynamicAU_RCNN_ROI_Extractor


class ActionUnitEvaluator(Evaluator):

    trigger = 1, 'epoch'
    default_name = 'AU_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, model, device, database, paper_report_label, converter, use_feature_map, sample_frame):
        super(ActionUnitEvaluator, self).__init__(iterator, model, device=device, converter=converter)
        self.T = sample_frame
        self.database = database
        self.use_feature_map = use_feature_map
        self.paper_use_AU = []
        self.paper_report_label = paper_report_label  # original AU_idx -> original AU
        paper_report_label_idx = list(paper_report_label.keys())  # original AU_idx

        self.AU_convert = dict()  # new_AU_idx -> AU
        for new_AU_idx, orig_AU_idx in enumerate(sorted(paper_report_label_idx)):
            self.AU_convert[new_AU_idx] = paper_report_label[orig_AU_idx]

        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA



    @overrides
    def evaluate(self):
        iterator = self._iterators['main']
        _target = self._targets["main"]
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)
        reporter = Reporter()
        reporter.add_observer("main", _target)
        summary = DictSummary()
        model = _target
        pred_labels_array = []
        gt_labels_array = []
        for idx, batch in enumerate(it):
            print("processing :{}".format(idx))
            batch = self.converter(batch, self.device)
            images, bboxes, labels = batch  # images shape = B*T, C, H, W; bboxes shape = B*T, F, 4; labels shape = B*T, F, 12
            if not isinstance(images, chainer.Variable):
                images = chainer.Variable(images.astype('f'))
                bboxes = chainer.Variable(bboxes.astype('f'))

            labels = labels.reshape(labels.shape[0] // self.T, self.T, labels.shape[1], labels.shape[2])
            mini_batch = labels.shape[0]
            if isinstance(model.au_rcnn_train_chain, DynamicAU_RCNN_ROI_Extractor):
                images = images.reshape(images.shape[0] // self.T, self.T, images.shape[1], images.shape[2],
                                        images.shape[3])
                bboxes = bboxes.reshape(bboxes.shape[0] // self.T, self.T, bboxes.shape[1], bboxes.shape[2])



            roi_feature = model.au_rcnn_train_chain(images, bboxes)  # shape =  B*T, F, C, H, W or  B*T, F, D
            if not self.use_feature_map:
                roi_feature = roi_feature.reshape(mini_batch, self.T, config.BOX_NUM[self.database], -1)  # shape = B, T, F, D
            else:
                # B*T*F, C, H, W => B, T, F, C, H, W
                roi_feature = roi_feature.reshape(mini_batch, self.T, config.BOX_NUM[self.database], roi_feature.shape[-3],
                                                  roi_feature.shape[-2], roi_feature.shape[-1])

            pred_labels = model.loss_head_module.predict(roi_feature)  # B, T, F, 12
            pred_labels = pred_labels[:, -1, :, :]  # B, F, D
            pred_labels = np.bitwise_or.reduce(pred_labels, axis=1)  # B, class_number
            labels = labels[:, -1, :, :]  # B, F, D
            labels = np.bitwise_or.reduce(chainer.cuda.to_cpu(labels), axis=1)  # B, class_number
            assert labels.shape == pred_labels.shape
            pred_labels_array.extend(pred_labels)
            gt_labels_array.extend(labels)

        gt_labels_array = np.stack(gt_labels_array)
        pred_labels_array = np.stack(pred_labels_array)  # shape = all_N, out_size

        gt_labels = np.transpose(gt_labels_array) # shape = Y x frame
        pred_labels = np.transpose(pred_labels_array) #shape = Y x frame
        report_dict = dict()
        AU_id_convert_dict = self.AU_convert if self.AU_convert else config.AU_SQUEEZE
        for new_AU_idx, frame_pred in enumerate(pred_labels):
            if AU_id_convert_dict[new_AU_idx] in self.paper_use_AU:
                AU = AU_id_convert_dict[new_AU_idx]
                frame_gt = gt_labels[new_AU_idx]
                F1 = f1_score(y_true=frame_gt, y_pred=frame_pred)
                report_dict[AU] = F1
                summary.add({"f1_frame_avg": F1})

        observation = {}
        with reporter.scope(observation):
            reporter.report(report_dict, model)
            reporter.report(summary.compute_mean(), model)
        return observation