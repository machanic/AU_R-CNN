import numpy as np
import cv2
from action_unit_metric.get_AUC import AUC
from collections_toolkit.dot_dict import Map
from sklearn.metrics import auc as sk_auc


def get_segs(labels):
    '''
    Get segments from binary labels
    '''
    labels = labels.copy()
    labels[np.where(labels!=1)] = 0
    labels = labels.astype(np.uint8)
    connect_arr = cv2.connectedComponents(labels, connectivity=8, ltype=cv2.CV_32S)  # mask shape = 1 x H x W
    component_num = connect_arr[0]
    label_matrix = connect_arr[1]
    segs = []
    n_segs = component_num - 1
    for component_label in range(1, component_num):
        row_col = np.where(label_matrix == component_label)[0]
        segs.append(row_col)
    return segs, n_segs

def get_F1_event(label, pred):
    '''
     Get F1-Event
    :param label: binary ground truth label
    :param pred:  prediction
    :return: met a structure of F1E metric
    '''
    gt_segs, n_gt_segs = get_segs(label)  # n_gt_segs is the number of true events in whole video sequence
    bin_pred = pred > 0
    bin_pred = bin_pred.astype(np.int32)
    pred_segs, n_pred_segs = get_segs(bin_pred)  # pred_segs is the number of detected events in the whole video sequence
    gt_frames = np.where(label>0)[0]
    pred_frames = np.where(pred > 0)[0]
    ths = np.arange(0, 1.01, 0.01)
    # init
    n_th = ths.size
    TPP = np.zeros(n_th) # TP for precision
    TPR = np.zeros(n_th) # TP for recall
    # Compute overlap score for GT
    olScoreGt = np.zeros(n_gt_segs)
    for i in range(0, n_gt_segs):
        seg = gt_segs[i]
        olScoreGt[i] = np.intersect1d(seg, pred_frames).size / seg.size  # each segment part have one calculate

    # Compute overlap score for predicted segments
    olScorePr = np.zeros(n_pred_segs)
    for i in range(n_pred_segs):
        seg = pred_segs[i]
        olScorePr[i] = np.intersect1d(seg, gt_frames).size / seg.size
    # compute TP and recall and precision
    for iOl in range(n_th):
        TPR[iOl] = np.sum(olScoreGt >= ths[iOl])
        TPP[iOl] = np.sum(olScorePr >= ths[iOl])
    ER = TPR / n_gt_segs  # ER is the ratio of correctly detected events over the true events.
    EP = TPP / n_pred_segs  # EP is the ratio of correctly detected events over the detected events.
    # Compute f1-Event
    F1E_curve = 2 * ER * EP / (ER + EP)

    # Compute AUC under the F1E curve
    x = np.concatenate((np.array([0]), ths, np.array([1])), axis=0)
    y = np.concatenate((np.array([1]), F1E_curve, np.array([0])))
    # auc = AUC(x, y)
    auc = sk_auc(x,y,reorder=False)
    if np.isnan(auc):
        raise ValueError("AUC is NaN,(no true event exists)")
    # get output
    met = Map()
    met.f1EventCurve = F1E_curve
    met.thresholds = ths
    met.TP_recall = TPR
    met.TP_precision = TPP
    met.olScoreGt = olScoreGt
    met.olScorePr = olScorePr
    met.nGtSeg = n_gt_segs
    met.nPrSeg = n_pred_segs
    met.auc = auc
    return met