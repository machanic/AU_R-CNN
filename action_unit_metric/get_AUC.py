import numpy as np
def AUC(x, y):
    ''' Compute Area Under the Curve (AUC) specified by coordinates in vectors x and y
    '''
    ind = np.argsort(x)
    x = np.sort(x)

    y = y[ind]
    xc = np.append(np.spacing(1), x.flatten())
    xc = np.append(xc, 1).T
    yc = np.append(np.spacing(1), y.flatten())
    yc = np.append(yc, 1).T
    xdiff = np.zeros(len(yc))
    for i in range(0, len(yc)-1):
        xdiff[i] = xc[i+1] - xc[i]

    auc = np.sum(xdiff*yc)
    return auc
