import copy

import chainer
import chainer.training.extensions
import numpy as np
from chainer import reporter
from chainer.training.extensions import Evaluator

import config
from AU_rcnn.legacy.eval_AU_label_occur import eval_AU_occur
from AU_rcnn.utils.bin_label_translate import AUbin_label_translate
from structural_rnn.model.open_crf.pure_python.constant_variable import LabelTypeEnum


def eval_func(nodeid_pred, node_id_gt_labels):

    pred_labels = []
    gt_labels = []
    for node_id, pred in nodeid_pred.items():
        pred_labels.append(AUbin_label_translate(pred))
        gt_labels.append(AUbin_label_translate(node_id_gt_labels[node_id]))
    return eval_AU_occur(pred_labels, gt_labels)

class S_RNN_Evaluator(Evaluator):

    trigger = 1, "epoch"
    default_name = "validation"
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, converter_func, device):
        super(S_RNN_Evaluator, self).__init__(iterator, target,
                                              converter=converter_func, device=device)


    def get_combine_int_label(self, label_dict, label_bin):
        label_str = []
        label_arr = np.nonzero(label_bin)[0]
        for label_squeeze_idx in label_arr:
            label_str.append(config.AU_SQUEEZE[label_squeeze_idx])
        label_str = ",".join(sorted(label_str))
        return label_dict.get_id_const(label_str)


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
        target = self._targets['main'] # target is StructureRNN
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
        label_dict = None
        summary = reporter.DictSummary()
        # it = self.converter(it, device=self.device)  # FIXME 这句要写converter
        for batch in it:
            xs, crf_pact_structures = self.converter(batch, device=self.device)
            xp = chainer.cuda.get_array_module(xs)
            with_crf = target.with_crf
            with chainer.no_backprop_mode():
                orig_pred_labels = target.predict(xs, crf_pact_structures)  # labels is one batch multiple videos' labels
            gt_labels_batch = target.get_gt_labels(np, crf_pact_structures, is_bin=False)  # gt_labels shape= B x N x D, B is batch_size
            orig_pred_labels = chainer.cuda.to_cpu(orig_pred_labels)
            gt_labels_batch = chainer.cuda.to_cpu(gt_labels_batch)
            compare_pred_labels = list()

            if not with_crf:
                for idx, crf_pact in enumerate(crf_pact_structures):
                    one_compare_pred_labels = list()
                    one_video_preds = orig_pred_labels[idx]
                    sample = crf_pact.sample
                    label_dict = sample.label_dict
                    for label_bin in one_video_preds:
                        label_int = self.get_combine_int_label(label_dict, label_bin)
                        one_compare_pred_labels.append(label_int)
                    compare_pred_labels.append(one_compare_pred_labels)
            else:
                compare_pred_labels = orig_pred_labels

            if cnt is None:  # crf_pact_structure.num_label 指的是bin类型的vector长度，而非算上各种组合的label_dict的长度
                cnt = np.zeros(shape=(len(label_dict), len(label_dict)), dtype=np.uint32)
            if ucnt is None:
                ucnt = np.zeros(shape=(len(label_dict), len(label_dict)), dtype=np.uint32)
            for idx, pred_labels in enumerate(compare_pred_labels):
                gt_labels = gt_labels_batch[idx]
                for i in range(len(pred_labels)):
                    if pred_labels[i] == gt_labels[i]:
                        hit += 1
                    else:
                        miss += 1
                    cnt[pred_labels[i], gt_labels[i]] += 1
                    if sample.node_list[i].label_type == LabelTypeEnum.UNKNOWN_LABEL:
                        if pred_labels[i] == gt_labels[i]:
                            hitu += 1
                        else:
                            missu += 1
                        ucnt[pred_labels[i], gt_labels[i]] += 1

        precision, recall, F1, accuracy = self.evaluate_criterion(len(label_dict), cnt)
        unkown_precision, unkown_recall, unkown_F1, unkown_accuracy = self.evaluate_criterion(len(label_dict),
                                                                                                  ucnt)
        report = {"A_hit": hit, "U_hit": hitu,
                  "A_miss": miss, "U_miss": missu,
                  "A_Accuracy": float(hit) / (hit + miss),
                  "U_Accuracy": float(hitu) / (hitu + missu),
                  "A_Precision": precision, "U_Precision": unkown_precision,
                  "A_Recall": recall, "U_Recall": unkown_recall,
                  "A_F1": F1, "U_F1": unkown_F1
                  }
        print(report)
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation



