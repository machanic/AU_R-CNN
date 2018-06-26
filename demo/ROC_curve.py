import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from scipy import interp

import numpy as np

import config


def read_AU_RCNN_npz(npz_path):
    npz_file = np.load(npz_path)
    pred_score = npz_file['pred_score']  # 41522, 9, 22
    gt_label = npz_file['gt']  # 41522, 9, 22
    pred_score = np.maximum.reduce(pred_score, axis=1, dtype=np.float32)
    gt_label = np.bitwise_or.reduce(gt_label, axis=1)
    return pred_score, gt_label

def read_CNN_npz(npz_path):
    npz_file = np.load(npz_path)
    pred_score = npz_file['pred_score']   # 41555, 22
    gt_label = npz_file['gt']  # 41555, 22
    return pred_score, gt_label

def get_roc_curve(pred_scores, gt_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(gt_labels.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(gt_labels[:, i], pred_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(gt_labels.ravel(), pred_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc, fpr, tpr

def plot_roc_curve(CNN_roc_auc, CNN_fpr, CNN_tpr, AURCNN_roc_auc, AURCNN_fpr, AURCNN_tpr):
    for AU_idx,AU in config.AU_SQUEEZE.items():
        if AU in config.paper_use_BP4D:
            plt.figure()
            lw = 2
            plt.plot(CNN_fpr[AU_idx], CNN_tpr[AU_idx], color='darkorange',
                     lw=lw, label='CNN ROC curve (area = %0.2f)' % CNN_roc_auc[AU_idx])
            plt.plot(AURCNN_fpr[AU_idx], AURCNN_tpr[AU_idx], color='green',
                     lw=lw, label='AU R-CNN ROC curve (area = %0.2f)' % AURCNN_roc_auc[AU_idx])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            plt.savefig('E:/AU_{}_ROC.eps'.format(AU), format='eps', dpi=1000)

def main():
    CNN_pred_score, CNN_gt_label = \
        read_CNN_npz("G:/Facial AU detection dataset/AU R-CNN trained file/CNN模型/CNN_result/npz_split_3.npz")
    AURCNN_pred_score, AURCNN_gt_label = \
        read_AU_RCNN_npz("G:/Facial AU detection dataset/AU R-CNN trained file/npz_BP4D_AU_RCNN/npz_split_3.npz")
    CNN_roc_auc, CNN_fpr, CNN_tpr = get_roc_curve(CNN_pred_score, CNN_gt_label)
    AURCNN_roc_auc, AURCNN_fpr, AURCNN_tpr = get_roc_curve(AURCNN_pred_score, AURCNN_gt_label)

    plot_roc_curve(CNN_roc_auc, CNN_fpr, CNN_tpr, AURCNN_roc_auc, AURCNN_fpr, AURCNN_tpr)

if __name__ == "__main__":
    main()