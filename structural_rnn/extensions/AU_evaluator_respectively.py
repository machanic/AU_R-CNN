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
        super(ActionUnitEvaluator, self).__init__(iterator, list(target_dict.values())[0], device=device)
        self.database = database
        self.target_dict =target_dict
        self.paper_use_AU = []
        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA
        elif database == "BP4D_DISFA":
            self.paper_use_AU = set(config.paper_use_BP4D + config.paper_use_DISFA)



    # 不合成，分别报告
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

        trainer_result = defaultdict(list)

        for batch in it:

            batch = convert(batch, self.device)
            for x, crf_pact_structure in zip(*batch):

                sample = crf_pact_structure.sample
                file_path = sample.file_path
                print("evaluate file:{0}".format(file_path))
                video_id = os.path.basename(file_path)
                train_keyword = os.path.basename(os.path.dirname(file_path))  # train_keyword comes from file_path

                if train_keyword not in self.target_dict:
                    print("error {} not pre-trained".format(train_keyword))
                    continue
                target = self.target_dict[train_keyword]  # choose the right predictor
                # pred_probs is N x (Y'+1)
                pred_labels = target.predict(x, crf_pact_structure, is_bin=False)  # pred_labels is  N x 1
                gt_labels = target.get_gt_label_one_graph(np, crf_pact_structure, is_bin=False)  # return N x 1
                trainer_result[train_keyword].append((video_id, pred_labels, gt_labels))


        report_dict = defaultdict(dict) # key = train_keyword
        for train_keyword, pred_gt_list in trainer_result.items():
            trainer_pred_labels = []
            trainer_gt_labels = []
            for video_id, pred_labels, gt_labels in sorted(pred_gt_list, key=lambda e:e[0]):
                trainer_pred_labels.extend(pred_labels)
                trainer_gt_labels.extend(gt_labels)
            trainer_pred_labels = np.asarray(trainer_pred_labels,dtype=np.int32)
            trainer_gt_labels = np.asarray(trainer_gt_labels,dtype=np.int32)

            assert len(trainer_gt_labels) == len(trainer_pred_labels)

            node_number = len(trainer_gt_labels)
            gt_labels = np.zeros((node_number, len(config.AU_SQUEEZE)), dtype=np.int32) # frame x Y
            pred_labels = np.zeros((node_number, len(config.AU_SQUEEZE)), dtype=np.int32)
            for node_idx, gt_label in enumerate(trainer_gt_labels):
                if gt_label > 0:
                    AU = train_keyword.split("_")[gt_label - 1]
                    AU_squeeze_idx = config.AU_SQUEEZE.inv[AU]
                    gt_labels[node_idx, AU_squeeze_idx] = 1
                pred_label = trainer_pred_labels[node_idx]
                if pred_label > 0:
                    AU = train_keyword.split("_")[pred_label - 1]
                    AU_squeeze_idx = config.AU_SQUEEZE.inv[AU]
                    pred_labels[node_idx, AU_squeeze_idx] = 1
            gt_labels = gt_labels.reshape(-1, config.BOX_NUM[self.database], len(config.AU_SQUEEZE))
            pred_labels = pred_labels.reshape(-1, config.BOX_NUM[self.database], len(config.AU_SQUEEZE))
            gt_labels = np.bitwise_or.reduce(gt_labels, axis=1) # shape = frame x Y
            pred_labels = np.bitwise_or.reduce(pred_labels, axis=1) # shape = frame x Y

            gt_labels = np.transpose(gt_labels) # shape = Y x frame
            pred_labels = np.transpose(pred_labels) #shape = Y x frame
            for AU_idx, frame_pred in enumerate(pred_labels):
                if config.AU_SQUEEZE[AU_idx] in self.paper_use_AU:
                    if config.AU_SQUEEZE[AU_idx] in train_keyword.split("_"):

                        AU = config.AU_SQUEEZE[AU_idx]
                        frame_gt = gt_labels[AU_idx]
                        F1 = f1_score(y_true=frame_gt, y_pred=frame_pred)
                        accuracy = accuracy_score(frame_gt, frame_pred)
                        report_dict[train_keyword][AU] = F1
        merge_dict = {}
        for train_keyword, AU_F1 in report_dict.items():
            for AU, F1 in AU_F1.items():
                if AU in self.paper_use_AU:
                    if AU not in merge_dict:
                        merge_dict[AU] = F1
                    elif F1 > merge_dict[AU]:
                        merge_dict[AU] = F1
        report_dict["merge_result"] = merge_dict
        observation = {}
        with reporter.scope(observation):
            reporter.report(report_dict, _target)
            reporter.report(summary.compute_mean(), _target)
        print(report_dict)
        return observation