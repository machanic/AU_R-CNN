import numpy as np

def norm_cm(cm):
    '''
    compute skew-normalized f1 score from confusion matrix
    :param cm: confusion matrix
    :return: f1n - skew-normalized f1 score
            pn - skew-normalized precision
            rn - skew-normalized recall
            ncm - skew-normalized confusion matrix
            s - the skewness factor
    '''
    num_pos = np.sum(cm[:, 0])
    num_neg = np.sum(cm[:, 1])
    if num_pos == 0:
        ncm = np.zeros((2,2),dtype=np.int32)
        print("no positive samples found")
        return ncm
    skew = num_neg/ float(num_pos)
    ncm = np.array([cm[:, 0], cm[:,1]/skew]).T
    return ncm, skew

def cm2f1n(cm):
    '''
    cm2f1n.m
    Compute skew-normalized f1 score from confusion matrix
    :param cm:
    :return:
    '''
    ncm, s = norm_cm(cm)
    f1n = 0
    pn = 0
    rn = 0
    if ncm[0,0] > 0: # TP has to be > 0
        pn = ncm[0,0]/ np.sum(ncm[0, :])
        rn = ncm[0,0]/ np.sum(ncm[:, 0])
        f1n = 2 * pn * rn / (pn + rn)
    return f1n, pn, rn, ncm, s


