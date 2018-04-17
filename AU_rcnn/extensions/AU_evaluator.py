import copy

import chainer.training.extensions
import numpy as np
from chainer import Reporter
from action_unit_metric.F1_frame import get_F1_frame
from action_unit_metric.get_ROC import get_ROC
from action_unit_metric.F1_event import get_F1_event
import config
from collections import defaultdict
from chainer import DictSummary
from sklearn.metrics import f1_score


class AUEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, "epoch"
    default_name = "AU_RCNN_validation"
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, concat_example_func, database, device):
        super(AUEvaluator, self).__init__(iterator, target, converter=concat_example_func,device=device)
        self.paper_use_AU = []
        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA
        elif database == "BP4D_DISFA":
            self.paper_use_AU = set(config.paper_use_BP4D + config.paper_use_DISFA)
        self.database = database

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']


        it = copy.copy(iterator)

        all_gt_label = []
        all_pred_label = []

        for idx, batch in enumerate(it):

            batch = self.converter(batch, device=self.device)
            imgs, bbox, labels = batch
            imgs = chainer.Variable(imgs)
            bbox = chainer.Variable(bbox)
            if bbox.shape[1] != config.BOX_NUM[self.database]:
                print("error box num {0} != {1}".format(bbox.shape[1], config.BOX_NUM[self.database]))
                continue
            preds, _ = target.predict(imgs, bbox)  # R', class_num
            preds = preds.reshape(labels.shape[0], labels.shape[1], labels.shape[2])  # shape = B, N, Y
            preds = chainer.cuda.to_cpu(preds)
            labels = chainer.cuda.to_cpu(labels)
            preds = np.bitwise_or.reduce(preds, axis=1)  # shape = B, Y
            gt_labels = np.bitwise_or.reduce(labels, axis=1) # shape = B, Y


            all_gt_index = set()
            pos_pred = np.nonzero(preds)
            pos_gt_labels = np.nonzero(gt_labels)
            all_gt_index.update(list(zip(*pos_pred)))
            all_gt_index.update(list(zip(*pos_gt_labels)))
            if len(all_gt_index) > 0:
                accuracy = np.sum(preds[list(zip(*all_gt_index))[0], list(zip(*all_gt_index))[1]] == gt_labels[list(zip(*all_gt_index))[0], list(zip(*all_gt_index))[1]])/ len(all_gt_index)
                print("batch idx:{0} current batch accuracy is :{1}".format(idx, accuracy))
            all_gt_label.extend(gt_labels)
            all_pred_label.extend(preds)
        all_gt_label = np.asarray(all_gt_label)  # shape = (N, len(AU_SQUEEZE))
        all_pred_label = np.asarray(all_pred_label)  # shape = (N, len(AU_SQUEEZE))
        AU_gt_label = np.transpose(all_gt_label)  # shape = (len(AU_SQUEEZE), N)
        AU_pred_label = np.transpose(all_pred_label)  # shape=  (len(AU_SQUEEZE), N)
        report = defaultdict(dict)

        reporter = Reporter()
        reporter.add_observer("main", target)
        summary = DictSummary()
        for AU_squeeze_idx, pred_label in enumerate(AU_pred_label):
            AU = config.AU_SQUEEZE[AU_squeeze_idx]
            if AU in self.paper_use_AU:
                gt_label = AU_gt_label[AU_squeeze_idx]
                # met_E = get_F1_event(gt_label, pred_label)
                met_F = get_F1_frame(gt_label, pred_label)
                roc =get_ROC(gt_label, pred_label)
                f1 = f1_score(gt_label, pred_label)
                report["f1_frame"][AU] = met_F.f1f
                report["f1_score"][AU] = f1
                assert f1 == met_F.f1f
                report["AUC"][AU] = roc.auc
                report["accuracy"][AU] = met_F.accuracy
                # report["f1_event"][AU] = np.median(met_E.f1EventCurve)
                summary.add({"f1_frame_avg": f1})
                summary.add({"AUC_avg": roc.auc})
                summary.add({"accuracy_avg": met_F.accuracy})
                # summary.add({"f1_event_avg": np.median(met_E.f1EventCurve)})
        observation = {}
        with reporter.scope(observation):
            reporter.report(report, target)
            reporter.report(summary.compute_mean(), target)
        return observation