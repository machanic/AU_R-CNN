import chainer
from overrides import overrides
from chainer import reporter
import copy
import numpy as np
import config
from collections import defaultdict
from action_unit_metric.F1_frame import get_F1_frame
from action_unit_metric.get_ROC import get_ROC
from action_unit_metric.F1_event import get_F1_event
from chainer.training.extensions import Evaluator


class ActionUnitEvaluator(Evaluator):
    trigger = 1, 'epoch'
    default_name = 'S_RNN_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, device, database):
        super(ActionUnitEvaluator, self).__init__(iterator, target, device=device)
        self.database = database
        self.paper_use_AU = []
        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA
        elif database == "BP4D_DISFA":
            self.paper_use_AU = set(config.paper_use_BP4D + config.paper_use_DISFA)

    @overrides
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']  # target is S_RNN_Plus
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)
        summary = reporter.DictSummary()
        label_num = len(config.AU_SQUEEZE)
        all_pred_batch = []
        all_gt_batch = []
        for batch in it:
            xs, crf_pact_structures = self.converter(batch, device=self.device)
            pred_labels_batch = []
            for x, crf_pact_structure in zip(xs, crf_pact_structures):
                pred_labels = target.predict(x, crf_pact_structure)  # pred_labels is  N x D, but open-crf predict only produce shape = N
                if pred_labels.ndim == 1:
                    AU_bins = []  # AU_bins is labels in one video sequence
                    for pred_label in pred_labels:  # pred_label is int id of combine AU. multiple AU combine seem as one
                        AU_list = self.label_dict.get_key(pred_label).split(",")  # actually, because Open-CRF only support single label prediction
                        AU_bin = np.zeros(len(config.AU_SQUEEZE), dtype=np.int32)
                        for AU in AU_list:
                            np.put(AU_bin, config.AU_SQUEEZE.inv[AU], 1)
                        AU_bins.append(AU_bin)
                    pred_labels = np.asarray(AU_bins)  # shape = N x D

                pred_labels_batch.append(pred_labels)

            gt_labels_batch = target.get_gt_labels(np, crf_pact_structures,
                                                   is_bin=True)    #  gt_labels shape= B x N x Y, B is batch_size, Y is label num
            all_pred_batch.append(pred_labels_batch)
            all_gt_batch.append(gt_labels_batch)
        pred_labels_batch = np.concatenate(np.asarray(all_pred_batch), axis=0)  # shape = T x N x D where T = B x B x B...
        gt_labels_batch = np.concatenate(np.asarray(all_gt_batch), axis=0)  # shape = T x N x D
        assert pred_labels_batch.shape[-1] == label_num  # shape = B x N x D
        assert gt_labels_batch.shape[-1] == label_num
        box_num = config.BOX_NUM[self.database]
        pred_labels_batch = pred_labels_batch.reshape(-1, box_num, label_num)  # shape = (B x Frame) x box_num x Y
        gt_labels_batch = gt_labels_batch.reshape(-1, box_num, label_num) # shape = (B x Frame) x box_num x Y
        pred_labels_batch = np.bitwise_or.reduce(pred_labels_batch, axis=1)
        gt_labels_batch = np.bitwise_or.reduce(gt_labels_batch, axis=1)

        gt_labels_batch = np.transpose(gt_labels_batch, (1,0)) #shape = Y x N. where N = (B x Frame)
        pred_labels_batch = np.transpose(pred_labels_batch, (1,0)) #shape = Y x N where N = (B x Frame)
        report = defaultdict(dict)
        for AU_idx, gt_label in enumerate(gt_labels_batch):
            AU = config.AU_SQUEEZE[AU_idx]
            if AU in self.paper_use_AU:
                pred_label = pred_labels_batch[AU_idx]
                met_E = get_F1_event(gt_label, pred_label)
                met_F = get_F1_frame(gt_label, pred_label)
                roc = get_ROC(gt_label, pred_label)
                AU = config.AU_SQUEEZE[AU_idx]
                report["f1_frame"][AU] = met_F.f1f
                report["AUC"][AU] = roc.auc
                report["f1_event"][AU] = np.median(met_E.f1EventCurve)
                summary.add({"f1_frame_avg": met_F.f1f})
                summary.add({"AUC_avg": roc.auc})
                summary.add({"f1_event_avg": np.median(met_E.f1EventCurve)})
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(summary, target)
            reporter.report(summary.compute_mean(), target)
        return observation