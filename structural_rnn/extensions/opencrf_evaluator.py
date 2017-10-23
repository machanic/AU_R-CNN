import copy

import chainer
import numpy as np
from chainer import reporter
from chainer.training.extensions import Evaluator

from structural_rnn.model.open_crf.pure_python.constant_variable import LabelTypeEnum


class OpenCRFEvaluator(Evaluator):

    trigger = 1, "epoch"
    default_name = "opencrf_val"
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, device):
        super(OpenCRFEvaluator, self).__init__(iterator, target,
                                               device=device)

    def evaluate_criterion(self, num_label:int, matrix:np.ndarray):
        precision = 0.0
        recall = 0.0
        F1 = 0.0
        accuracy = 0.0
        for y in range(num_label):
            FN = 0
            FP = 0
            TN = 0
            diag_val = matrix[y,y]
            TP = diag_val
            sum_column = 0
            sum_row = 0
            for i in range(num_label):
                for j in range(num_label):
                    if i == y:
                        sum_row += matrix[i, j]
                    if j == y:
                        sum_column += matrix[i, j]
                    if i != y and j != y:
                        TN += matrix[i, j]
            FP = sum_row - diag_val
            FN = sum_column - diag_val
            precision += float(TP) / (TP + FP)
            recall += float(TP) /(TP + FN)
            F1 += 2 * float(TP) / (2*TP + FP + FN)
            accuracy += (float(TP) + float(TN)) / (TP + FP + TN + FN)
        precision /= float(num_label)
        recall /= float(num_label)
        F1 /= float(num_label)
        accuracy /= float(num_label)
        return precision,recall,F1,accuracy

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)
        hit = 0
        miss = 0
        hitu = 0
        missu = 0
        cnt = None
        ucnt = None

        for batch in it:
            # batch = self.converter(batch, device=self.device)
            for x, crf_pact_structure in batch:
                sample = crf_pact_structure.sample
                pred_labels = target.predict(x, crf_pact_structure)
                gt_labels = []
                if cnt is None:
                    cnt = np.zeros(shape=(crf_pact_structure.num_label, crf_pact_structure.num_label), dtype=np.uint32)
                if ucnt is None:
                    ucnt = np.zeros(shape=(crf_pact_structure.num_label, crf_pact_structure.num_label), dtype=np.uint32)
                for node in sample.node_list:
                    gt_labels.append(node.label)

                for i in range(len(sample.node_list)):
                    if pred_labels[i] == gt_labels[i]:
                        hit += 1
                    else:
                        miss += 1
                    cnt[pred_labels[i], sample.node_list[i].label] += 1
                    if sample.node_list[i].label_type == LabelTypeEnum.UNKNOWN_LABEL:
                        if pred_labels[i] == sample.node_list[i].label:
                            hitu += 1
                        else:
                            missu += 1
                        ucnt[pred_labels[i], sample.node_list[i].label] += 1
        # print("A_HIT={0} U_HIT={1}".format(hit, hitu))
        # print("A_MISS={0} U_MISS={1}".format(miss, missu))
        precision, recall, F1, accuracy = self.evaluate_criterion(crf_pact_structure.num_label, cnt)
        # unkown_precision, unkown_recall, unkown_F1, unkown_accuracy = self.evaluate_criterion(crf_pact_structure.num_label, ucnt)
        report = { "hit":hit, #"U_hit":hitu,
                   "miss":miss, #"U_miss":missu,
                   "accuracy" : float(hit)/ (hit+ miss),
                   # "U_Accuracy" : float(hitu)/ (hitu+missu),
                   "A_Precision": precision,# "U_Precision":unkown_precision,
                   "A_Recall":recall,# "U_Recall":unkown_recall,
                   "F1": F1, #"U_F1":unkown_F1
                   }
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
