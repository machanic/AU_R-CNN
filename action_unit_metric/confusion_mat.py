import numpy as np

# Regulate non-positive values to -1
def reg(y):
    yout = np.int8(np.sign(y.flatten()))
    yout[np.where(yout!=1)] = -1
    return yout



def confmat(label, pred):
    '''
    Compute confusion matrix:
    cm = | TP FP |
         | FN TN |
    :param label:  ground truth label
    :param pred:  prediction
    '''
    pred = reg(pred).reshape(label.shape)
    TP = np.sum((pred==1) & (label==1))
    TN = np.sum((pred!=1) & (label!=1))
    FP = np.sum((pred==1) & (label!=1))
    FN = np.sum((pred!=1) & (label==1))
    cm = np.array([[TP, FP],[FN, TN]])
    return cm