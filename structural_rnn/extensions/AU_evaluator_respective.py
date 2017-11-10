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


    # 目标：把AU_idx对应的跨越frame的序列搞出来
    # 1. 合成多个trainer的prob挑出最大的，合并到一个prob里
    # 2. 合成多个gt_label的文件夹中的ground truth信息，合并到一个视频的node里
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

        video_gt_bin_dict = defaultdict(list) # key = video_id, value = gt_bin of different
        video_pred_prob_dict = defaultdict(list)

        for batch in it:

            batch = convert(batch, self.device)
            for x, crf_pact_structure in zip(*batch):

                sample = crf_pact_structure.sample
                file_path = sample.file_path
                print("evaluate file:{}".format(file_path))
                video_id = os.path.basename(file_path)
                train_keyword = os.path.basename(os.path.dirname(file_path))  # train_keyword comes from file_path

                if train_keyword not in self.target_dict:
                    print("error {} not pre-trained".format(train_keyword))
                    continue
                target = self.target_dict[train_keyword]  # choose the right predictor
                # pred_probs is N x (Y'+1)
                _, pred_probs = target.predict(x, crf_pact_structure, is_bin=True)  # pred_labels is  N x Y', but open-crf predict only produce shape = N
                gt_labels = target.get_gt_label_one_graph(np, crf_pact_structure, is_bin=False)  # return N x Y'

                gt_bins = [] # N list of each is AU_SQUEEZE len
                for gt_label in gt_labels: # for-loop N times, N is number of nodes
                    gt_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
                    if gt_label > 0:
                        AU = train_keyword.split("_")[gt_label - 1]
                        AU_idx = config.AU_SQUEEZE.inv[AU]
                        gt_bin[AU_idx] = 1
                    gt_bins.append(gt_bin)
                pred_bins = []
                for pred_prob in pred_probs:
                    pred_bin = np.zeros(shape=len(config.AU_SQUEEZE)+1, dtype=np.float32)  # shape = Y +1  , index = 0 store all 0 prob
                    for pred_idx, pred_score in enumerate(pred_prob):
                        if pred_idx > 0:
                            AU = train_keyword.split("_")[pred_idx-1]
                            AU_idx = config.AU_SQUEEZE.inv[AU]
                            pred_bin[AU_idx + 1] = pred_score
                        else:
                            pred_bin[0] = pred_score
                    pred_bins.append(pred_bin)
                video_pred_prob_dict[video_id].append(pred_bins) # list of N x (Y'+1)
                video_gt_bin_dict[video_id].append(gt_bins) # list of N x Y , this list is append cross trainer

        pred_final = []
        gt_final = []
        for video_id, pred_probs in video_pred_prob_dict.items():
            pred_probs = np.asarray(pred_probs)  # shape = U x N x (Y' + 1)
            pred_probs = np.transpose(pred_probs, (1,0,2))  # shape = N x U x (Y' + 1)
            pred_probs_index = np.argmax(pred_probs, axis=2)  # shape = N x U , 这U个里面只存一个数字，指示哪个Y'最大
            pred_probs_max_score = np.max(pred_probs, axis=2)  # shape = N x U , 这U个里面只存一个数字，最大的Y'的分数
            pred_probs_max_trainer = np.argmax(pred_probs_max_score, axis=1) # shape = N , 这N个里面只存一个数字，指示最大的U是谁
            pred_labels = pred_probs_index[np.arange(pred_probs_index.shape[0]), pred_probs_max_trainer]  # shape = N
            # 但是此时pred_labels 是一个shape=N的每个元素是0~Y'的，0表示都是0，1表示bins的第0位是1
            pred_bins_array = np.zeros(shape=(pred_labels.shape[0], pred_probs.shape[2]-1), dtype=np.int32) # shape = N x Y'
            for pred_idx, pred_label in enumerate(pred_labels):
                if pred_label > 0:
                    pred_bins_array[pred_idx, pred_label-1] = 1

            gt_labels = np.asarray(video_gt_bin_dict[video_id]) # U x N x Y'
            gt_labels = np.transpose(gt_labels, axes=(1,0,2)) # N x U x Y'
            gt_labels = np.bitwise_or.reduce(gt_labels, axis=1) # N x Y' after elementwise or, #FIXME这句话在这里写错了，是跨越多个不同trainer的同一个video，gt应该不变，但是不是跨越多个trainer
            pred_final.append(pred_bins_array)
            gt_final.append(gt_labels)
        pred_final = np.concatenate(pred_final, axis=0)  # shape = N' x Y'
        gt_final = np.concatenate(gt_final, axis=0)      #shape = N' x Y'

        pred_final = pred_final.reshape(-1, config.BOX_NUM[self.database], pred_final.shape[1]) # shape = Frame x box x Y'
        gt_final = gt_final.reshape(-1, config.BOX_NUM[self.database], gt_final.shape[1]) # shape = Frame x box x Y'
        pred_final = np.bitwise_or.reduce(pred_final, axis=1)  # shape =Frame x Y'
        gt_final = np.bitwise_or.reduce(gt_final, axis=1)  # shape = Frame x Y' after element or
        pred_final = np.transpose(pred_final)
        gt_final = np.transpose(gt_final)


        #         pred_bins = []  # pred_bins is labels in one video sequence
        #         gt_bins = [] # gt_bins is labels in one video sequence
        #         pred_prob_bins = []
        #         for idx, pred_prob in enumerate(pred_probs): # N times iterator, N is number of nodes
        #             pred_prob_bin = np.zeros(shape=(len(config.AU_SQUEEZE)+1), dtype=np.float32) # Y + 1 because pred=0 also have prob
        #             for pred_idx in range(pred_prob.shape[0]):
        #                 if pred_idx == 0:
        #                     pred_prob_bin[0] = pred_prob[0]  # 第0位置上表示全都是0
        #                 else:
        #                     AU = train_keyword.split("_")[pred_idx - 1]
        #                     AU_idx = config.AU_SQUEEZE.inv[AU]
        #                     pred_prob_bin[AU_idx + 1] = pred_prob[pred_idx]
        #             pred_prob_bins.append(pred_prob_bin) # list of Y + 1
        #
        #         for gt_label in gt_labels: # N times iterator, N is number of nodes
        #             gt_bin = np.zeros(shape=len(config.AU_SQUEEZE), dtype=np.int32)  # shape = Y
        #             if gt_label > 0:
        #                 AU = train_keyword.split("_")[gt_label - 1]
        #                 AU_idx = config.AU_SQUEEZE.inv[AU]
        #                 gt_bin[AU_idx] = 1
        #             gt_bins.append(gt_bin)
        #
        #         pred_bins = np.asarray(pred_bins)  # shape = N x Y (Y is AU_squeeze length)
        #         gt_bins = np.asarray(gt_bins)
        #         pred_prob_bins = np.asarray(pred_prob_bins)
        #         assert len(pred_bins) == len(sample.node_list)
        #         assert len(gt_bins) == len(sample.node_list)
        #         video_pred_bin_dict[video_id].append(pred_bins)  # each pred_bins is shape = N x Y. but N of each graph is different
        #         video_gt_bin_dict[video_id].append(gt_bins)  # each gt_bins is shape = N x Y. but N of each graph is different
        #         video_pred_prob_dict[video_id].append(pred_prob_bins)  # each pred_probs = N x (Y+1)
        # assert len(video_gt_bin_dict) == len(video_pred_bin_dict)
        # # predict final is determined by vote
        # video_pred_final = []  # shape = list of  N x Y ,each N is different in each video
        # video_gt_final = []   # shape = list of N x Y, each N is different
        # for video_id, pred_prob_bins in video_pred_prob_dict.items():
        #     prod_prob_bins_array = np.asarray(pred_prob_bins)  # shape = U x N x (Y+1) , where U is different trainer number, this time N is the same cross diferent video
        #     prod_prob_bins_array = np.transpose(prod_prob_bins_array,(1,0,2))  # shape = N x U x (Y+1)
        #     prod_prob_bins_index = np.argmax(prod_prob_bins_array, axis=2)  # shape = N x U choose the biggest Y index in last axis
        #     prod_prob_bins_array = np.max(prod_prob_bins_array, axis=2)  # shape = N x U. each element is prob number
        #     choice_trainer_index = np.argmax(prod_prob_bins_array, axis=1)  # shape = N, each element is which U is biggest
        #     pred_labels = prod_prob_bins_index[np.arange(len(prod_prob_bins_index)), choice_trainer_index]  #shape = N, each element is correct Y
        #
        #     pred_bins_array = np.zeros(shape=(pred_labels.shape[0], len(config.AU_SQUEEZE)),dtype=np.int32)
        #     for pred_idx, pred_label in enumerate(pred_labels):
        #         if pred_label != 0:
        #             pred_bins_array[pred_idx, pred_label - 1] = 1
        #     video_pred_final.append(pred_bins_array) # list of N x Y
        #
        #     # for gt_label part, we don't need vote, we only need element-wise or
        #     gt_bins_array = np.asarray(video_gt_bin_dict[video_id])  #  # shape = U x N x Y , where U is different trainer number
        #     gt_bins_array = np.transpose(gt_bins_array, axes=(1,0,2)) # shape = N x U x Y
        #     video_gt_final.append(np.bitwise_or.reduce(gt_bins_array, axis=1))  # list shape = N x Y
        #
        # video_pred_final = np.concatenate(video_pred_final, axis=0) # shape = N' x Y ,where N' is total nodes of all frames cross videos
        # video_gt_final = np.concatenate(video_gt_final, axis=0)  # shape = N' x Y ,where N' is total nodes of all frames cross videos
        # box_num = config.BOX_NUM[self.database]
        # # we suppose that the n nodes order as frame, each frame have 9/8 boxes
        # pred_labels_batch = video_pred_final.reshape(-1, box_num, len(config.AU_SQUEEZE)) # shape = (V x Frame) x box_num x Y
        # gt_labels_batch = video_gt_final.reshape(-1, box_num, len(config.AU_SQUEEZE)) # shape = (V x Frame) x box_num x Y
        # pred_labels_batch = np.bitwise_or.reduce(pred_labels_batch, axis=1)  # shape = (V x Frame) x Y
        # gt_labels_batch = np.bitwise_or.reduce(gt_labels_batch, axis=1) # shape = (V x Frame) x Y
        #
        # gt_labels_batch = np.transpose(gt_labels_batch, (1,0)) #shape = Y x N'. where N' = (V x Frame)
        # pred_labels_batch = np.transpose(pred_labels_batch, (1,0)) #shape = Y x N' where N' = (V x Frame)
        report = defaultdict(dict)
        for gt_idx, gt_label in enumerate(gt_final):

            pred_label = pred_final[gt_idx]
            # met_E = get_F1_event(gt_label, pred_label)
            F1 = f1_score(y_true=gt_label, y_pred=pred_label)
            accuracy = accuracy_score(gt_label, pred_label)
            met_F = get_F1_frame(gt_label, pred_label)
            # roc = get_ROC(gt_label, pred_label)
            report["f1_frame"][gt_idx] = met_F.f1f
            # report["AUC"][AU] = roc.auc
            report["accuracy"][gt_idx] = accuracy
            summary.add({"f1_frame_avg": F1})
            # summary.add({"AUC_avg": roc.auc})
            summary.add({"accuracy_avg": accuracy})
        observation = {}
        with reporter.scope(observation):
            reporter.report(report, _target)
            reporter.report(summary.compute_mean(), _target)
        print(observation)
        return observation