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
import os

class ActionUnitEvaluator(Evaluator):

    trigger = 1, 'epoch'
    default_name = 'S_RNN_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target_dict, device, database):
        super(ActionUnitEvaluator, self).__init__(iterator, list(target_dict.keys())[0], device=device) #FIXME random pick one target
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
        _target = self._iterators["target"]
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
        N = 0  # N is node number in one video, but each 9 nodes composite of a box
        for batch in it:

            if self.device >=0:
                x = [chainer.cuda.to_gpu(x) for x, _ in batch]
            else:
                x = [chainer.cuda.to_cpu(x) for x, _ in batch]
            crf_pact_structures =[p for _, p in batch]
            batch = [x, crf_pact_structures]
            for x, crf_pact_structure in zip(*batch):
                if N == 0:
                    N = x.shape[0]
                assert N == x.shape[0]
                sample = crf_pact_structure.sample
                file_path = sample.file_path
                train_AU_labels = os.path.basename(os.path.dirname(file_path))
                video_id = os.path.basename(file_path)
                target = self.target_dict[train_AU_labels]  # choose the right predictor
                pred_labels = target.predict(x, crf_pact_structure, is_bin=False)  # pred_labels is  N x D, but open-crf predict only produce shape = N
                assert N == pred_labels.shape[0]
                gt_labels = target.get_gt_label_one_batch(np, crf_pact_structure, is_bin=True)  # return N x D
                assert N == gt_labels.shape[0]
                assert pred_labels.ndim == 1
                pred_bins = []  # pred_bins is labels in one video sequence
                gt_bins = [] # gt_bins is labels in one video sequence
                for pred_label in pred_labels:
                    pred_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    if pred_label > 0:
                        AU = train_AU_labels.split("_")[pred_label - 1]
                        AU_idx = config.AU_SQUEEZE.inv[AU]
                        pred_bin[AU_idx] = 1  # CRF can only predict one label, translate to AU_squeeze length
                    pred_bins.append(pred_bin)
                for gt_label in gt_labels:
                    assert len(gt_label) == len(train_AU_labels.split("_"))
                    gt_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    non_zero_gt_idx = np.nonzero(gt_label)[0]
                    if len(non_zero_gt_idx) > 0:
                        for gt_idx in non_zero_gt_idx:
                            AU = train_AU_labels.split("_")[gt_idx - 1]
                            AU_idx = config.AU_SQUEEZE.inv[AU]
                            gt_bin[AU_idx] = 1  # translate to AU_squeeze length
                    gt_bins.append(gt_bin)

                pred_bins = np.asarray(pred_bins)  # shape = N x Y (Y is AU_squeeze length)
                gt_bins = np.asarray(gt_bins)
                assert len(pred_bins) == sample.n
                assert len(gt_bins) == sample.n
                video_pred_bin_dict[video_id].append(pred_bins)  # each pred_bins is shape = N x Y
                video_gt_bin_dict[video_id].append(gt_bins)  # each gt_bins is shape = N x Y
        assert len(video_gt_bin_dict) == len(video_pred_bin_dict)
        # predict final is determined by vote
        video_pred_final = np.zeros(shape=(len(video_gt_bin_dict), N, len(config.AU_SQUEEZE)))  # shape = V x N x Y , where V is video number
        video_gt_final = np.zeros(shape=(len(video_gt_bin_dict), N, len(config.AU_SQUEEZE)))   # shape = V x N x Y
        for video_idx, (video_id, pred_bins_list) in sorted(video_pred_bin_dict.items(), key=lambda e:e[0]):
            pred_bins_array = np.asarray(pred_bins_list)  # shape = U x N x Y , where U is different trainer number
            count_array = np.zeros(shape=(pred_bins_array.shape[1], pred_bins_array.shape[2], 2), dtype=np.int32)  # N x Y x 2 (+1/-1) for vote
            for pred_bins in pred_bins_array: # pred_bins_array shape = U x N x Y
                for n, pred_bin in enumerate(pred_bins):  # pred_bins shape = N x Y
                    for pred_idx, pred_val in enumerate(pred_bin): # pred_bin shape = Y
                        count_array[n, pred_idx, pred_val] += 1
            video_pred_final[video_idx] = np.argmax(count_array, axis=2) # shape = N x Y

            # for gt_label part, we don't need vote, we only need element-wise or
            gt_bins_array = np.asarray(video_gt_bin_dict[video_id])  #  # shape = U x N x Y , where U is different trainer number
            gt_bins_array = np.transpose(gt_bins_array, axes=(1,0,2)) # shape = N x U x Y
            video_gt_final[video_idx] = np.bitwise_or.reduce(gt_bins_array, axis=1)  # shape = N x Y


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
            reporter.report(report, _target)
            reporter.report(summary.compute_mean(), _target)
        return observation