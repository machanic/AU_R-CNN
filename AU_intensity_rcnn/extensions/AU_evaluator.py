import copy
from collections import defaultdict

import chainer.training.extensions
import numpy as np
from chainer import DictSummary
from chainer import Reporter
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error

import config


class AUEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, "epoch"
    default_name = "AU_RCNN_validation"
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self,  iterator, target, concat_example_func, database, output_dir, device):
        super(AUEvaluator, self).__init__(iterator, target, converter=concat_example_func,device=device)
        self.paper_use_AU = []
        if database == "BP4D":
            self.paper_use_AU = config.paper_use_BP4D
        elif database == "DISFA":
            self.paper_use_AU = config.paper_use_DISFA
        elif database == "BP4D_DISFA":
            self.paper_use_AU = set(config.paper_use_BP4D + config.paper_use_DISFA)
        self.database = database
        self.output_dir = output_dir

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        it = copy.copy(iterator)

        all_gt_label = []
        all_score = []
        for idx, batch in enumerate(it):
            print("processing idx: {}".format(idx))
            batch = self.converter(batch, device=self.device)
            imgs, bbox, labels = batch
            if imgs is None:
                continue
            xp = chainer.cuda.get_array_module(imgs)

            imgs = chainer.Variable(imgs)
            bbox = chainer.Variable(bbox)
            if bbox.shape[1] != config.BOX_NUM[self.database]:
                print("error box num {0} != {1}".format(bbox.shape[1], config.BOX_NUM[self.database]))
                continue
            scores = target.predict(imgs, bbox)  # R', class_num
            scores = scores.reshape(labels.shape[0], labels.shape[1], labels.shape[2])  # shape = B,F,Y
            labels = chainer.cuda.to_cpu(labels)  # B, F, Y
            scores = chainer.cuda.to_cpu(scores)  # B, F, Y
            labels = np.maximum.reduce(labels, axis=1)  # B, Y
            scores = np.maximum.reduce(scores, axis=1)  # B, Y
            all_gt_label.extend(labels)
            all_score.extend(scores)

        all_gt_label = np.asarray(all_gt_label)  # shape = (N, 5)
        all_score = np.asarray(all_score)  # shape = (N, 5)

        all_gt_label = np.transpose(all_gt_label)  # 5, N
        all_score = np.transpose(all_score)  # 5, N

        report = defaultdict(dict)

        reporter = Reporter()
        reporter.add_observer("main", target)
        summary = DictSummary()
        for idx, score in enumerate(all_score):
            AU = config.AU_INTENSITY_DICT[idx]
            gt_label = all_gt_label[idx]
            error = mean_squared_error(gt_label, score)
            pearson_correlation, _ = pearsonr(gt_label,
                                              score)
            report["mean_squared_error"][AU] = error
            report["pearson_correlation"][AU] = pearson_correlation

            summary.add({"pearson_correlation_avg": pearson_correlation})
            summary.add({"mean_squared_error_avg": error})

        observation = {}
        with reporter.scope(observation):
            reporter.report(report, target)
            reporter.report(summary.compute_mean(), target)

        return observation