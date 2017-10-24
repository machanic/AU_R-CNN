import numpy as np
import warnings
def cm2f1f(cm):
    '''
    cm2f1f.m
    :param cm: 2 x 2 confusion matrix
    cm = | TP FP |
         | FN TN |
    :return:
        f1f - frame-based f1
        p - precision
        r - recall
    '''
    f1f = np.nan
    p = 0
    r = 0
    accuracy = 0
    if cm[0,0] > 0:  # TP has to be >0, otherwise just return 0
        p = cm[0,0] / np.sum(cm[0, :])   # precision = TP / (TP + FP)
        r = cm[0,0] / np.sum(cm[:, 0])   # recall = TP / (TP + FN)
        f1f = 2 * p * r / (p + r)
        accuracy = (cm[0,0] + cm[1,1])/np.sum(cm)
    else:
        warnings.warn("error the f1 frame call pass cm TP <= 0!")
    return f1f, p, r,accuracy