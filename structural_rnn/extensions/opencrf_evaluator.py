import copy

import chainer
import numpy as np
from chainer import reporter
from chainer.training.extensions import Evaluator

from structural_rnn.model.open_crf.pure_python.constant_variable import LabelTypeEnum
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

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
        it = copy.copy(iterator)
        hit = 0
        miss = 0
        hitu = 0
        missu = 0
        cnt = None
        ucnt = None
        pred_labels = []
        gt_labels = []
        for batch in it:
            if self.device >=0:
                x = [chainer.cuda.to_gpu(x) for x,_ in batch]
            else:
                x = [chainer.cuda.to_cpu(x) for x,_ in batch]
            pact =[p for _, p in batch]
            batch = [x, pact]
            for x, crf_pact_structure in zip(*batch):
                sample = crf_pact_structure.sample
                pred_label = target.predict(x, crf_pact_structure, is_bin=False)
                pred_labels.extend(pred_label)
                gt_label = []
                # if cnt is None:
                #     cnt = np.zeros(shape=(crf_pact_structure.num_label, crf_pact_structure.num_label), dtype=np.uint32)
                # if ucnt is None:
                #     ucnt = np.zeros(shape=(crf_pact_structure.num_label, crf_pact_structure.num_label), dtype=np.uint32)

                for i, node in enumerate(sample.node_list):
                    gt_label.append(node.label)
                    if pred_label[i] == gt_label[i]:
                        hit += 1
                    else:
                        miss+=1
                gt_labels.extend(gt_label)

                    # cnt[pred_labels[i], sample.node_list[i].label] += 1
                    # if sample.node_list[i].label_type == LabelTypeEnum.UNKNOWN_LABEL:
                    #     if pred_labels[i] == sample.node_list[i].label:
                    #         hitu += 1
                    #     else:
                    #         missu += 1
                    #     ucnt[pred_labels[i], sample.node_list[i].label] += 1

        # print("A_HIT={0} U_HIT={1}".format(hit, hitu))
        # print("A_MISS={0} U_MISS={1}".format(miss, missu))
        # precision, recall, F1, accuracy = self.evaluate_criterion(crf_pact_structure.num_label, cnt)
        assert len(gt_labels) == len(pred_labels)
        F1 = f1_score(y_true=gt_labels, y_pred=pred_labels, average='weighted')
        precision = precision_score(y_true=gt_labels, y_pred=pred_labels, average='weighted')
        accuracy = accuracy_score(y_true=gt_labels, y_pred=pred_labels)
        recall = recall_score(y_true=gt_labels, y_pred=pred_labels,average='weighted')
        # unkown_precision, unkown_recall, unkown_F1, unkown_accuracy = self.evaluate_criterion(crf_pact_structure.num_label, ucnt)
        report = { "hit":hit, #"U_hit":hitu,
                   "miss":miss, #"U_miss":missu,
                   # "accuracy" : float(hit)/ (hit+ miss),
                   "accuracy" : accuracy,
                   # "U_Accuracy" : float(hitu)/ (hitu+missu),
                   "precision": precision,# "U_Precision":unkown_precision,
                   "recall":recall,# "U_Recall":unkown_recall,
                   "F1": F1, #"U_F1":unkown_F1
                   }
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
