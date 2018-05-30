import copy
import os

import chainer
import chainer.functions as F
import numpy as np
from chainer import DictSummary
from chainer import Reporter
from chainer.training.extensions import Evaluator
from overrides import overrides
from sklearn.metrics import f1_score

import config


class ActionUnitEvaluator(Evaluator):

    trigger = 1, 'epoch'
    default_name = 'AU_validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, model, device, database, paper_report_label, converter,
                 output_path):
        super(ActionUnitEvaluator, self).__init__(iterator, model, device=device, converter=converter)
        self.database = database
        self.paper_use_AU = []
        self.paper_report_label = paper_report_label  # original AU_idx -> original AU
        paper_report_label_idx = list(paper_report_label.keys())  # original AU_idx
        self.output_path = output_path
        self.AU_convert = dict()  # new_AU_idx -> AU
        for new_AU_idx, orig_AU_idx in enumerate(sorted(paper_report_label_idx)):
            self.AU_convert[new_AU_idx] = paper_report_label[orig_AU_idx]
        self.class_num = len(self.AU_convert)

        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA



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
        reporter.add_observer("main", _target)
        summary = DictSummary()
        model = _target
        pred_labels_array = []
        gt_labels_array = []

        unreduce_pred = []
        unreduce_gt = []

        last_seq_id = None
        one_frame_predict_list = []
        one_frame_gt_list = []

        for idx, batch in enumerate(it):
            print("processing :{}".format(idx))
            batch = self.converter(batch, self.device)
            feature, gt_seg_rgb, gt_seg_flow, seg_info, seg_labels, gt_labels, npz_file_path = batch  # feature shape =(B, C, W); bboxes shape = B,W 4; labels shape B, W, 12
            seg_id = os.path.basename(npz_file_path[0])
            seg_id = seg_id[:seg_id.rindex("#")]
            if last_seq_id is None:
                last_seq_id = seg_id

            if last_seq_id != seg_id:
                one_frame_predict_result = np.stack(one_frame_predict_list, axis=2)  # B, W, F, class
                unreduce_pred.extend(one_frame_predict_result.reshape(-1,
                                                              one_frame_predict_result.shape[-2],one_frame_predict_result.shape[-1]))  # list of W, F, class
                one_frame_predict_result = np.bitwise_or.reduce(one_frame_predict_result, axis=2) # B, W, class
                one_frame_predict_result = one_frame_predict_result.reshape([-1, one_frame_predict_result.shape[-1]]) # B* W, class
                pred_labels_array.extend(one_frame_predict_result)

                one_frame_gt_result = np.stack(one_frame_gt_list, axis=2)  # B, W, F, class
                unreduce_gt.extend(one_frame_gt_result.reshape(-1, one_frame_gt_result.shape[-2], one_frame_gt_result.shape[-1])) # list of W, F, class
                one_frame_gt_result = np.bitwise_or.reduce(one_frame_gt_result, axis=2) # B, W, class
                one_frame_gt_result = one_frame_gt_result.reshape([-1, one_frame_gt_result.shape[-1]])  # B * W, class
                gt_labels_array.extend(one_frame_gt_result)

                one_frame_predict_list.clear()
                one_frame_gt_list.clear()

            if not isinstance(feature, chainer.Variable):
                feature = chainer.Variable(feature.astype('f'))
             # feature = (B, C, W)
            predict_labels = model.predict(feature)  # (B, W, class)
            one_frame_predict_list.append(predict_labels)

            gt_labels = chainer.cuda.to_cpu(gt_labels)
            one_frame_gt_list.append(gt_labels)

        one_frame_predict_result = np.stack(one_frame_predict_list, axis=2)  # B, W, F, class
        unreduce_pred.extend(one_frame_predict_result.reshape(-1,
                                                              one_frame_predict_result.shape[-2],one_frame_predict_result.shape[-1]))  # list of W, F, class
        one_frame_predict_result = np.bitwise_or.reduce(one_frame_predict_result, axis=2)  # B, W, class
        assert one_frame_predict_result.shape[-1] == self.class_num
        one_frame_predict_result = one_frame_predict_result.reshape(
            [-1, one_frame_predict_result.shape[-1]])  # B* W, class
        pred_labels_array.extend(one_frame_predict_result)

        one_frame_gt_result = np.stack(one_frame_gt_list, axis=2)  # B, W, F, class
        unreduce_gt.extend(one_frame_gt_result.reshape(-1, one_frame_gt_result.shape[-2], one_frame_gt_result.shape[-1]))  # list of W, F, class
        one_frame_gt_result = np.bitwise_or.reduce(one_frame_gt_result, axis=2)  # B, W, class
        one_frame_gt_result = one_frame_gt_result.reshape([-1, one_frame_gt_result.shape[-1]])  # B * W, class
        gt_labels_array.extend(one_frame_gt_result)

        one_frame_predict_list.clear()
        one_frame_gt_list.clear()


        # 由于F不一样，因此不能stack
        # unreduce_pred = np.stack(unreduce_pred).astype(np.int32)  # N, W, F, class
        # unreduce_gt = np.stack(unreduce_gt).astype(np.int32)  # N, W, F, class
        # np.savez(self.output_path , predict=unreduce_pred, gt=unreduce_gt)

        gt_labels_array = np.stack(gt_labels_array)  # all_N, 12
        pred_labels_array = np.stack(pred_labels_array)  # shape = all_N, out_size
        gt_labels = np.transpose(gt_labels_array) # shape = Y x frame
        pred_labels = np.transpose(pred_labels_array) #shape = Y x frame
        report_dict = dict()
        AU_id_convert_dict = self.AU_convert if self.AU_convert else config.AU_SQUEEZE
        for new_AU_idx, frame_pred in enumerate(pred_labels):
            if AU_id_convert_dict[new_AU_idx] in self.paper_use_AU:
                AU = AU_id_convert_dict[new_AU_idx]
                frame_gt = gt_labels[new_AU_idx]
                F1 = f1_score(y_true=frame_gt, y_pred=frame_pred)
                report_dict[AU] = F1
                summary.add({"f1_frame_avg": F1})

        observation = {}
        with reporter.scope(observation):
            reporter.report(report_dict, model)
            reporter.report(summary.compute_mean(), model)
        return observation