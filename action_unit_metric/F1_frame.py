from action_unit_metric.confusion_mat import reg,confmat
from action_unit_metric.confusion_mat_f1_frame import cm2f1f
from collections_toolkit.dot_dict import Map

def get_F1_frame(label, pred):
    '''
    Compute F1-Frame
    :param label: binary ground turth, type is np.ndarray
    :param pred: prediction
    :return: met  - a structure of F1F metric
    '''
    label = reg(label)
    pred = reg(pred)
    # Compute confusion mat and f1
    cm = confmat(label, pred)
    f1f, p, r, accuracy = cm2f1f(cm)
    # packing
    met = Map()
    met.f1f = f1f
    met.p = p
    met.r = r
    met.cm = cm
    met.accuracy = accuracy
    return met
