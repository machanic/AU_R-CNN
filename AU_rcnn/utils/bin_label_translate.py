import config
import numpy as np
def AUbin_label_translate(bin_array):
    '''
    from 0,1,0,1,0 --> "1", "5", where "1" and "5" are actual FACS AU number
    :param bin_array:
    :return:
    '''
    return [config.AU_SQUEEZE[i] for i in np.where(bin_array > 0)[0]]
