from action_unit_metric.confusion_mat import reg,confmat
from action_unit_metric.confusion_mat_f1_normalize import cm2f1n
from collections_toolkit.dot_dict import Map


def get_F1_norm(label, pred):
    '''
    Compute F1-Norm
    Input:
    label   - binary ground turth label
    pred    - prediction

    Output:
    met     - a structure of F1N metric
    '''
    label = reg(label)
    pred = reg(pred)
    # Compute confusion mat and f1
    cm = confmat(label, pred)
    f1n, pn, rn, ncm, s = cm2f1n(cm)
    met = Map()
    met.f1n = f1n
    met.pn = pn
    met.rn = rn
    met.ncm = ncm
    met.s = s
    return met
