import numpy as np
from collections_toolkit.dot_dict import Map
from action_unit_metric.get_AUC import AUC
from action_unit_metric.confusion_mat import confmat
def get_ROC(label, pred):
    '''
    Compute the Receiver Operating Characteristics (ROC) curve
    :param label: binary ground turth label
    :param pred: prediction
    :return: met - a structure of roc metric
    '''
    pred = pred.flatten()
    label = label.flatten()
    met = Map()
    # Check if there is no ture positive
    if np.sum(label) == 0:
        met.auc = np.NaN
        met.roc_x = np.array([])
        met.roc_y = np.array([])
        return met

    assert pred.size == label.size, "pred and label size mismatch"
    assert np.unique(label).size <= 2, "label has to be binary for now"

    # Compute ROC
    vals = np.unique(pred)
    roc = np.zeros((len(vals),2))
    POS = label == 1;
    NEG = label == -1;
    for i in range(len(vals)):
        predpos = pred > vals[i]
        predneg = pred <= vals[i]
        TP = np.sum(POS & predpos)
        FP = np.sum(NEG & predpos)
        FN = np.sum(POS & predneg)
        TN = np.sum(NEG & predneg)
        TPR = 0
        FPR = 0
        if TP + FN != 0:
            TPR = float(TP) / (TP + FN)
        if TN + FP != 0:
            FPR = float(FP)/(TN + FP)
        roc[i, :] = [TPR, FPR]
    # Compute AUC
    auc = AUC(roc[:, 1], roc[:, 0])
    if np.isnan(auc):
        raise ValueError("AUC is NaN")
    met.auc = auc
    met.rocx = roc[:, 1]
    met.rocy = roc[:, 0]
    met.confmat = confmat(label, pred)
    return met