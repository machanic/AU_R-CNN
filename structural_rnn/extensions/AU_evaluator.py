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
from structural_rnn.updater.bptt_updater import convert


class ActionUnitEvaluator(Evaluator):

    trigger = 1, 'epoch'
    default_name = 'S_RNN_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target_dict, device, database):
        super(ActionUnitEvaluator, self).__init__(iterator, list(target_dict.values())[0], device=device)  # FIXME is it ok
        self.database = database
        self.target_dict =target_dict
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
        _target = self._targets["main"]
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)
        reporter = Reporter()
        reporter.add_observer("main", _target) # will fail to run?
        summary = DictSummary()

        video_gt_bin_dict = defaultdict(list) # key = video_id, value = predict bin list
        video_pred_bin_dict = defaultdict(list)  # key = video_id, value = predict bin list
        video_pred_prob_dict = defaultdict(list)

        for batch in it:

            batch = convert(batch, self.device)
            for x, crf_pact_structure in zip(*batch):

                sample = crf_pact_structure.sample
                file_path = sample.file_path
                train_keyword = os.path.basename(os.path.dirname(file_path))  # train_keyword comes from file_path
                video_id = os.path.basename(file_path)
                if train_keyword not in self.target_dict:
                    print("error {} not pre-trained".format(train_keyword))
                    continue
                target = self.target_dict[train_keyword]  # choose the right predictor
                # pred_probs is N x (Y+1)
                pred_labels, pred_probs = target.predict(x, crf_pact_structure, is_bin=False)  # pred_labels is  N x 1, but open-crf predict only produce shape = N
                gt_labels = target.get_gt_label_one_graph(np, crf_pact_structure, is_bin=False)  # return N x 1
                assert pred_labels.ndim == 1
                pred_bins = []  # pred_bins is labels in one video sequence
                gt_bins = [] # gt_bins is labels in one video sequence
                prod_prob_bins = []
                for idx, pred_label in enumerate(pred_labels): # N times iterator, N is number of nodes
                    pred_prob = pred_probs[idx]  # probability is len = Y+1
                    pred_prob_bin = np.zeros(shape=(len(config.AU_SQUEEZE)+1), dtype=np.float32) # Y + 1 because pred=0 also have prob
                    for pred_idx in range(pred_prob.shape[0]):
                        if pred_idx == 0:
                            pred_prob_bin[0] = pred_prob[0]  # 第0位置上表示全都是0
                        else:
                            AU = train_keyword.split("_")[pred_idx - 1]
                            AU_idx = config.AU_SQUEEZE.inv[AU]
                            pred_prob_bin[AU_idx + 1] = pred_prob[pred_idx]
                    prod_prob_bins.append(pred_prob_bin) # list of Y + 1

                    pred_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    if pred_label > 0:
                        AU = train_keyword.split("_")[pred_label - 1]
                        AU_idx = config.AU_SQUEEZE.inv[AU]
                        pred_bin[AU_idx] = 1  # CRF can only predict one label, translate to AU_squeeze length
                    pred_bins.append(pred_bin)
                for gt_label in gt_labels: # N times iterator, N is number of nodes
                    gt_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    if gt_label > 0:
                        AU = train_keyword.split("_")[gt_label - 1]
                        AU_idx = config.AU_SQUEEZE.inv[AU]
                        gt_bin[AU_idx] = 1
                    gt_bins.append(gt_bin)

                pred_bins = np.asarray(pred_bins)  # shape = N x Y (Y is AU_squeeze length)
                gt_bins = np.asarray(gt_bins)
                prod_prob_bins = np.asarray(prod_prob_bins)
                assert len(pred_bins) == len(sample.node_list)
                assert len(gt_bins) == len(sample.node_list)
                video_pred_bin_dict[video_id].append(pred_bins)  # each pred_bins is shape = N x Y. but N of each graph is different
                video_gt_bin_dict[video_id].append(gt_bins)  # each gt_bins is shape = N x Y. but N of each graph is different
                video_pred_prob_dict[video_id].append(prod_prob_bins)  # each pred_probs = N x (Y+1)
        assert len(video_gt_bin_dict) == len(video_pred_bin_dict)
        # predict final is determined by vote
        video_pred_final = []  # shape = list of  N x Y ,each N is different in each video
        video_gt_final = []   # shape = list of N x Y, each N is different
        for video_id, prod_prob_bins in video_pred_prob_dict.items():
            prod_prob_bins_array = np.asarray(prod_prob_bins)  # shape = U x N x (Y+1) , where U is different trainer number, this time N is the same cross diferent video
            prod_prob_bins_array = np.transpose(prod_prob_bins_array,(1,0,2))  # shape = N x U x (Y+1)
            prod_prob_bins_index = np.argmax(prod_prob_bins_array, axis=2)  # shape = N x U choose the biggest Y index in last axis
            prod_prob_bins_array = np.max(prod_prob_bins_array, axis=2)  # shape = N x U. each element is prob number
            choice_trainer_index = np.argmax(prod_prob_bins_array, axis=1)  # shape = N, each element is which U is biggest
            pred_labels = prod_prob_bins_index[np.arange(len(prod_prob_bins_index)), choice_trainer_index]  #shape = N, each element is correct Y

            pred_bins_array = np.zeros(shape=(pred_labels.shape[0], len(config.AU_SQUEEZE)),dtype=np.int32)
            for pred_idx, pred_label in enumerate(pred_labels):
                if pred_label != 0:
                    pred_bins_array[pred_idx, pred_label - 1] = 1
            video_pred_final.append(pred_bins_array) # list of N x Y

            # for gt_label part, we don't need vote, we only need element-wise or
            gt_bins_array = np.asarray(video_gt_bin_dict[video_id])  #  # shape = U x N x Y , where U is different trainer number
            gt_bins_array = np.transpose(gt_bins_array, axes=(1,0,2)) # shape = N x U x Y
            video_gt_final.append(np.bitwise_or.reduce(gt_bins_array, axis=1))  # list shape = N x Y

        video_pred_final = np.concatenate(video_pred_final, axis=0) # shape = N' x Y ,where N' is total nodes of all frames cross videos
        video_gt_final = np.concatenate(video_gt_final, axis=0)  # shape = N' x Y ,where N' is total nodes of all frames cross videos
        box_num = config.BOX_NUM[self.database]
        # we suppose that the n nodes order as frame, each frame have 9/8 boxes
        pred_labels_batch = video_pred_final.reshape(-1, box_num, len(config.AU_SQUEEZE)) # shape = (V x Frame) x box_num x Y
        gt_labels_batch = video_gt_final.reshape(-1, box_num, len(config.AU_SQUEEZE)) # shape = (V x Frame) x box_num x Y
        pred_labels_batch = np.bitwise_or.reduce(pred_labels_batch, axis=1)  # shape = (V x Frame) x Y
        gt_labels_batch = np.bitwise_or.reduce(gt_labels_batch, axis=1) # shape = (V x Frame) x Y

        gt_labels_batch = np.transpose(gt_labels_batch, (1,0)) #shape = Y x N'. where N' = (V x Frame)
        pred_labels_batch = np.transpose(pred_labels_batch, (1,0)) #shape = Y x N' where N' = (V x Frame)
        report = defaultdict(dict)
        for gt_idx, gt_label in enumerate(gt_labels_batch):
            AU = config.AU_SQUEEZE[gt_idx]
            if AU in self.paper_use_AU:
                pred_label = pred_labels_batch[gt_idx]
                # met_E = get_F1_event(gt_label, pred_label)
                F1 = f1_score(y_true=gt_label, y_pred=pred_label)
                accuracy = accuracy_score(gt_label, pred_label)
                met_F = get_F1_frame(gt_label, pred_label)
                # roc = get_ROC(gt_label, pred_label)
                report["f1_frame"][AU] = met_F.f1f
                # report["AUC"][AU] = roc.auc
                report["accuracy"][AU] = accuracy
                summary.add({"f1_frame_avg": F1})
                # summary.add({"AUC_avg": roc.auc})
                summary.add({"accuracy_avg": accuracy})
        observation = {}
        with reporter.scope(observation):
            reporter.report(report, _target)
            reporter.report(summary.compute_mean(), _target)
        print(observation)
        return observation