import scipy.io as sio
from action_unit_metric.get_ROC import get_ROC
from action_unit_metric.F1_event import get_F1_event
from action_unit_metric.F1_frame import get_F1_frame
import matplotlib.pyplot as plt
import numpy as np
from action_unit_metric.F1_norm import get_F1_norm
import matplotlib
from matplotlib.pyplot import savefig

def demo(mat_file_path):
    '''
    Facial action unit event detection by cascade of tasks
    '''
    result = sio.loadmat(mat_file_path)
    label = result['label'].reshape(-1).astype(np.int32)
    decV = result['decV'].reshape(-1).astype(np.float32)
    mask = label != 0
    label = label[mask]
    decV = decV[mask]
    metR = get_ROC(label, decV)

    metF = get_F1_frame(label, decV)

    # Compute f1-norm
    metN = get_F1_norm(label, decV)

    metE = get_F1_event(label, decV)

    # plot input
    plt.figure(figsize=(9, 7), dpi=200)

    plt.axis('tight')
    # plt.gca().set_position([80,80,1000,500])
    X = np.linspace(-10, 10, 4780, endpoint=True)
    plt.plot(X, label, linewidth=1.0, linestyle="-", label="label")
    plt.plot(X, decV,color="red", linewidth=0.6, linestyle="-", label='pred')
    plt.ylim(-4.0,4.0)
    plt.xlim(-10.0,10.0)
    plt.legend(loc='upper left')
    plt.plot([1, decV.size], [0, 0], linestyle=':',color='k')
    plt.title("Input signals")
    plt.xlabel('Frame index')
    plt.ylabel('Decision value')
    plt.show()
    # plot curve
    plt.close('all')
    plt.figure(figsize=(300, 200), dpi=800)
    f, ax = plt.subplots(2, sharex=True)
    ax[0].grid(color='r', linestyle='-', linewidth=0.1) # grid on
    ax[0].set_title('ROC curve (AUC={:.2f})'.format(metR.auc*100))
    ax[0].plot(metR.rocx, metR.rocy, color='b', linewidth=5)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")

    ax[1].set_title('F1-event curve (AUC={:.2f})'.format(metE.auc*100))
    ax[1].grid(color='r', linestyle='-', linewidth=0.1)  # grid on
    ax[1].plot(metE.thresholds, metE.f1EventCurve, color='r', linewidth=5)
    ax[1].set_xlabel('Overlap threshold')
    ax[1].set_ylabel("F1-Event")
    # ax[1].set_position([120,120,1200,500])
    plt.show()

if __name__ == "__main__":
    demo("D:/work/face_expr/data/test.mat")
