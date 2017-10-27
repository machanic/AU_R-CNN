import chainer
from overrides import overrides
from chainer import Reporter
from chainer import DictSummary
import copy
import numpy as np
import config
from collections import defaultdict
from action_unit_metric.F1_frame import get_F1_frame
from action_unit_metric.get_ROC import get_ROC
from action_unit_metric.F1_event import get_F1_event
from chainer.training.extensions import Evaluator
import json
from collections import OrderedDict

class ActionUnitEvaluator(Evaluator):
    trigger = 1, 'epoch'
    default_name = 'S_RNN_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, device, database, data_info_path):
        super(ActionUnitEvaluator, self).__init__(iterator, target, device=device)
        self.database = database
        self.paper_use_AU = []
        self.label_bin_len = 0
        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA
        elif database == "BP4D_DISFA":
            self.paper_use_AU = set(config.paper_use_BP4D + config.paper_use_DISFA)

        with open(data_info_path, "r") as file_obj:
            self.info_json = json.loads(file_obj.read())


    @overrides
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']  # target is S_RNN_Plus
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)
        reporter = Reporter()
        reporter.add_observer("main", target)
        summary = DictSummary()

        all_pred_batch = []
        all_gt_batch = []
        for batch in it:

            if self.device >=0:
                x = [chainer.cuda.to_gpu(x) for x,_ in batch]
            else:
                x = [chainer.cuda.to_cpu(x) for x,_ in batch]
            crf_pact_structures =[p for _, p in batch]
            batch = [x, crf_pact_structures]
            pred_labels_batch = []

            for x, crf_pact_structure in zip(*batch):
                label_bin_len = crf_pact_structure.label_bin_len
                self.label_bin_len = label_bin_len
                pred_labels = target.predict(x, crf_pact_structure, is_bin=False)  # pred_labels is  N x D, but open-crf predict only produce shape = N
                assert pred_labels.ndim == 1
                if pred_labels.ndim == 1: # 这个if必然会进去
                    AU_bins = []  # AU_bins is labels in one video sequence
                    for pred_label in pred_labels:
                        pred_bin = np.zeros(shape=label_bin_len, dtype=np.int32)  # shape = D
                        if pred_label >= 1:  # pred_label == 0 will be all zero, CRF
                            pred_bin[pred_label-1] = 1  # CRF 只能预测一个1
                        AU_bins.append(pred_bin)
                    pred_labels = np.asarray(AU_bins)  # shape = N x D (D is json_info file use_label_idx number)

                pred_labels_batch.append(pred_labels)  # shape B x N x D

            gt_labels_batch = target.get_gt_labels(np, crf_pact_structures,
                                                   is_bin=True)    #  gt_labels shape= B x N x D, B is batch_size, D is json_info file use_label_idx number
            assert gt_labels_batch.shape[-1] == self.label_bin_len  # label_bin_len length will be D
            all_pred_batch.append(pred_labels_batch)  # T x B x N x D (where T is iteration number)
            all_gt_batch.append(gt_labels_batch)  # T x B x N x D (where T is iteration number)
        pred_labels_batch = np.concatenate(np.asarray(all_pred_batch), axis=0)  # shape = T' x N x D where T' = B x B x B...
        gt_labels_batch = np.concatenate(np.asarray(all_gt_batch), axis=0)  # shape = T' x N x D
        assert pred_labels_batch.shape[-1] == self.label_bin_len
        assert gt_labels_batch.shape[-1] == self.label_bin_len
        box_num = config.BOX_NUM[self.database]
        pred_labels_batch = pred_labels_batch.reshape(-1, box_num, self.label_bin_len)  # shape = (B x Frame) x box_num x D
        gt_labels_batch = gt_labels_batch.reshape(-1, box_num, self.label_bin_len) # shape = (B x Frame) x box_num x D
        pred_labels_batch = np.bitwise_or.reduce(pred_labels_batch, axis=1)
        gt_labels_batch = np.bitwise_or.reduce(gt_labels_batch, axis=1)

        gt_labels_batch = np.transpose(gt_labels_batch, (1,0)) #shape = D x N. where N = (B x Frame)
        pred_labels_batch = np.transpose(pred_labels_batch, (1,0)) #shape = D x N where N = (B x Frame)
        report = defaultdict(dict)
        for gt_idx, gt_label in enumerate(gt_labels_batch):
            AU = config.AU_SQUEEZE[gt_idx]
            assert AU in self.paper_use_AU, "AU:{0} not in paper_use_AU:{1}".format(AU, self.paper_use_AU)
            if AU in self.paper_use_AU:
                pred_label = pred_labels_batch[gt_idx]
                # met_E = get_F1_event(gt_label, pred_label)
                met_F = get_F1_frame(gt_label, pred_label)
                # roc = get_ROC(gt_label, pred_label)
                report["f1_frame"][AU] = met_F.f1f
                # report["AUC"][AU] = roc.auc
                report["accuracy"][AU] = met_F.accuracy
                summary.add({"f1_frame_avg": met_F.f1f})
                # summary.add({"AUC_avg": roc.auc})
                summary.add({"accuracy_avg": met_F.accuracy})
        observation = {}
        with reporter.scope(observation):
            reporter.report(report, target)
            reporter.report(summary.compute_mean(), target)
        return observation