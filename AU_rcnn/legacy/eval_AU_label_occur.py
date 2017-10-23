from __future__ import division


import numpy as np

from collections import defaultdict
import six



def construct_down_sample_dict(gt_AU_sample_indices, is_down_sample):
    labels_down_sampled = defaultdict(dict)
    flatten = lambda l: np.array([item for sublist in l for item in sublist])
    for AU, sample_indices in gt_AU_sample_indices.items():  # AU comes from gt_label
        sample_indices = np.array(list(sample_indices))
        non_sample_indices = np.array(flatten([list(other_sample_indices_lst) \
                                               for other_AU, other_sample_indices_lst in gt_AU_sample_indices.items()
                                                if other_AU != AU]))
        non_sample_count = len(non_sample_indices)

        skew = float(non_sample_count) / len(sample_indices)
        labels_down_sampled[AU]["skew"] = skew # skew is stored in gt_labels_down_sampled
        if skew > 2.0 and is_down_sample: # neg class > pos class
            choice_number = len(sample_indices)
            choice_indices = np.random.choice(non_sample_indices, choice_number, replace=False)
            labels_down_sampled[AU]["pos"] = sample_indices
            labels_down_sampled[AU]["neg"] = choice_indices
        elif skew < 0.8 and is_down_sample:
            choice_number = non_sample_count
            choice_indices = np.random.choice(sample_indices, choice_number, replace=False)
            labels_down_sampled[AU]["pos"] = choice_indices
            labels_down_sampled[AU]["neg"] = non_sample_indices
        else:
            labels_down_sampled[AU]["pos"] = sample_indices
            labels_down_sampled[AU]["neg"] = non_sample_indices
    return labels_down_sampled


def random_downsample(pred_labels, gt_labels, is_down_sample):
    '''
    this legacy should be called after all batch all images LABEL read
    random_downsample should be down sample from all image in all batch not in particular image
    :param pred_labels: [[1,2,3], [2,5,6], [2,3]], AU 是不是这个数字，需要检查之前的传入的代码
    :param gt_labels: [[1,2,3,4,5], [2,3,4,5,6], [2,3]]
    :return:
    '''
    gt_AU_sample_indices = defaultdict(set)
    pred_AU_sample_indices = defaultdict(set)
    gt_contain_sub_list = all(isinstance(gt_label_image, list) for gt_label_image in gt_labels)
    pred_contain_sub_list = all(isinstance(pred_label_image, list) for pred_label_image in pred_labels)

    if gt_contain_sub_list:
        for img_index, gt_label_image in enumerate(gt_labels):
            for gt_label in gt_label_image:
                gt_AU_sample_indices[int(gt_label)].add(img_index)
    else:
        for img_index, gt_label in enumerate(gt_labels):
            gt_AU_sample_indices[int(gt_label)].add(img_index)

    if pred_contain_sub_list:
        for img_index, pred_label_image in enumerate(pred_labels):
            for pred_label in pred_label_image:
                pred_AU_sample_indices[int(pred_label)].add(img_index)
    else:
        for img_index, pred_label in enumerate(pred_labels):
            pred_AU_sample_indices[int(pred_label)].add(img_index)

    gt_labels_down_sampled = construct_down_sample_dict(gt_AU_sample_indices, is_down_sample)
    pred_labels_down_sampled = construct_down_sample_dict(pred_AU_sample_indices, is_down_sample)
    return pred_labels_down_sampled, gt_labels_down_sampled





def metric_F1(pred_labels, gt_labels, is_down_sample=False):
    pred_labels, gt_labels = random_downsample(pred_labels, gt_labels, is_down_sample=is_down_sample)
    report = dict()
    for AU in gt_labels.keys():  # 可能会有一些AU压根没有在测试集出现过
        confusion_matrix = np.zeros((2,2), dtype=np.float32)  # see: https://www.douban.com/note/284051363/?start=0&post=ok#last
        gt_pos_sample_indices = set(gt_labels[AU]["pos"].tolist())
        gt_neg_sample_indices = set(gt_labels[AU]["neg"].tolist())
        pred_pos_sample_indices = set(list(pred_labels[AU].get("pos", [])))
        pred_neg_sample_indices = set(list(pred_labels[AU].get("neg", [])))
        skew = gt_labels[AU]["skew"]
        for sample_index in gt_pos_sample_indices:  # confusion matrix first column
            confusion_matrix[0,0] += (sample_index in pred_pos_sample_indices)
            confusion_matrix[1,0] += (sample_index in pred_neg_sample_indices)
        for sample_index in gt_neg_sample_indices:
            confusion_matrix[0,1] += (sample_index in pred_pos_sample_indices)
            confusion_matrix[1,1] += (sample_index in pred_neg_sample_indices)
        TP = confusion_matrix[0,0]
        FP = confusion_matrix[0,1]
        FN = confusion_matrix[1,0]
        TN = confusion_matrix[1, 1]
        precision = TP/ float(TP + FP)
        recall = TP/float(TP + FN)

        F1 = 2 * precision * recall / (precision + recall)
        F1_norm = 2 * skew * precision * recall/(skew * recall + precision)
        report[AU] = dict()
        report[AU]["F1"] = F1
        report[AU]["F1_norm"] = F1_norm
        report[AU]["recall"] = recall
        report[AU]["precision"] = precision

    return report


def eval_AU_occur(pred_labels, gt_labels):
    '''

    :param pred_labels:  即可传入list of list [[1,2,3],[3,4,5]] 也可传入[1,2,3,4] 代表一幅图的所有label
    :param gt_labels: 即可传入list of list [[1,2,3],[3,4,5]] 也可传入[1,2,3,4] 代表一幅图的所有label
    :return:
    '''

    report = metric_F1(pred_labels, gt_labels, is_down_sample=False)
    _temp_report = dict()
    _temp_report["all"] = dict()
    _temp_report["all"]["F1"] = float(sum(report[AU]["F1"] for AU in report.keys())) / len(report.keys())
    _temp_report["all"]["F1_norm"] = float(sum(report[AU]["F1_norm"] for AU in report.keys())) / len(report.keys())
    _temp_report["all"]["precision"] = float(sum(report[AU]["precision"] for AU in report.keys())) / len(report.keys())
    _temp_report["all"]["recall"] = float(sum(report[AU]["recall"] for AU in report.keys())) / len(report.keys())
    report.update(_temp_report)
    return report