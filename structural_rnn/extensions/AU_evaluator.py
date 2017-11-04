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
                pred_labels = target.predict(x, crf_pact_structure, is_bin=False)  # pred_labels is  N x D, but open-crf predict only produce shape = N
                gt_labels = target.get_gt_label_one_graph(np, crf_pact_structure, is_bin=True)  # return N x D
                assert pred_labels.ndim == 1
                pred_bins = []  # pred_bins is labels in one video sequence
                gt_bins = [] # gt_bins is labels in one video sequence
                for pred_label in pred_labels:
                    pred_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    if pred_label > 0:
                        AU = train_keyword.split("_")[pred_label - 1]
                        AU_idx = config.AU_SQUEEZE.inv[AU]
                        pred_bin[AU_idx] = 1  # CRF can only predict one label, translate to AU_squeeze length
                    pred_bins.append(pred_bin)
                for gt_label_bin in gt_labels:
                    assert len(gt_label_bin) == len(train_keyword.split("_"))
                    gt_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    non_zero_gt_idx = np.nonzero(gt_label_bin)[0]
                    if len(non_zero_gt_idx) > 0:
                        for gt_idx in non_zero_gt_idx:
                            AU = train_keyword.split("_")[gt_idx]  # note that we didn't -1 here , because it is not predict value
                            AU_idx = config.AU_SQUEEZE.inv[AU]
                            gt_bin[AU_idx] = 1  # translate to AU_squeeze length
                    gt_bins.append(gt_bin)

                pred_bins = np.asarray(pred_bins)  # shape = N x Y (Y is AU_squeeze length)
                gt_bins = np.asarray(gt_bins)
                assert len(pred_bins) == len(sample.node_list)
                assert len(gt_bins) == len(sample.node_list)
                video_pred_bin_dict[video_id].append(pred_bins)  # each pred_bins is shape = N x Y. but N of each graph is different
                video_gt_bin_dict[video_id].append(gt_bins)  # each gt_bins is shape = N x Y. but N of each graph is different
        assert len(video_gt_bin_dict) == len(video_pred_bin_dict)
        # predict final is determined by vote
        video_pred_final = []  # shape = list of  N x Y ,each N is different
        video_gt_final = []   # shape = list of N x Y, each N is different
        for video_id, pred_bins_list in sorted(video_pred_bin_dict.items(), key=lambda e:e[0]):
            assert len(video_gt_bin_dict[video_id]) == len(pred_bins_list), (len(pred_bins_list), len(video_gt_bin_dict[video_id]))
            pred_bins_array = np.asarray(pred_bins_list)  # shape = U x N x Y , where U is different trainer number, this time N is the same cross diferent video
            count_array = np.zeros(shape=(pred_bins_array.shape[1], pred_bins_array.shape[2], 2), dtype=np.int32)  # N x Y x 2 (+1/-1) for vote
            for pred_bins in pred_bins_array: # pred_bins_array shape = U x N x Y
                for n, pred_bin in enumerate(pred_bins):  # pred_bins shape = N x Y
                    for pred_idx, pred_val in enumerate(pred_bin): # pred_bin shape = Y
                        count_array[n, pred_idx, pred_val] += 1
            video_pred_final.append(np.argmax(count_array, axis=2)) # list of shape = N x Y, because in video_pred_final, each N is different, thus we need list.append

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