import copy
import os
from collections import defaultdict

import chainer
import numpy as np
from chainer import DictSummary
from chainer import Reporter
from chainer.training.extensions import Evaluator
from overrides import overrides
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import config
from action_unit_metric.F1_frame import get_F1_frame


class ActionUnitEvaluator(Evaluator):

    trigger = 1, 'epoch'
    default_name = 'AU_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, model, device, database, paper_report_label, converter):
        super(ActionUnitEvaluator, self).__init__(iterator, model, device=device, converter=converter)
        self.database = database
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
        elif database == "BP4D_DISFA":
            self.paper_use_AU = set(config.paper_use_BP4D + config.paper_use_DISFA)



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
            feature, boxes, labels = batch  # each is B,T,F,D
            if not isinstance(feature, chainer.Variable):
                feature= chainer.Variable(feature)

            labels = np.bitwise_or.reduce(chainer.cuda.to_cpu(labels)[:,-1,:,:], axis=1)  # B, class_number
            pred_labels = model.predict(feature) # B, T, F, D -> B, class_number
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