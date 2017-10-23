import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class AU_LSTM(chainer.Chain):

    def __init__(self, in_size, out_size):
        super(AU_LSTM, self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1, in_size,out_size,0.5)
        self.neg_pos_ratio = 1

    def __call__(self, xs):  # xs shape = B x box_num x T x D
        if not isinstance(xs, chainer.Variable):
            xs = chainer.Variable(xs)
        batch_size = xs.shape[0]
        box_num = xs.shape[1]
        T = xs.shape[2]
        xs = xs.reshape(-1, xs.shape[2], xs.shape[3])

        xs = [x for x in xs]
        _,_,y_lst =self.lstm(None, None,xs)  # y_lst is list of T x Y ( Y is label_num)
        return F.stack(y_lst).reshape(batch_size,box_num, T, -1)  # return B x T x Y

    def predict(self, x): # x is shape = T x D inside one video graph
        ys = self.__call__([x])
        ys = chainer.cuda.to_cpu(ys)
        return (ys.data>0).astype(np.int32)

    def accuracy_func(self,ys,ts):
        xp = chainer.cuda.get_array_module(ys.data)  # ys and ts are batch = 1 of T x D
        ts = ts.reshape(-1, ts.shape[-1])
        ys = ys.reshape(-1, ys.shape[-1])  # shape = (box_num x T) x D
        union_gt = set()  # union of prediction positive and ground truth positive
        ts_data = ts.data if isinstance(ts, chainer.Variable) else ts
        cpu_gt_label = chainer.cuda.to_cpu(ts_data)  # shape = T x D
        gt_pos_index = np.nonzero(cpu_gt_label)
        cpu_pred_score = (chainer.cuda.to_cpu(ys.data) > 0).astype(np.int32)
        pred_pos_index = np.nonzero(cpu_pred_score)
        gt_pos_index_set = set(list(zip(*gt_pos_index)))
        pred_pos_index_set = set(list(zip(*pred_pos_index)))
        union_gt.update(gt_pos_index_set)
        union_gt.update(pred_pos_index_set)

        if len(union_gt) == 0:
            accuracy_pick_index = np.where(cpu_gt_label)
        else:
            accuracy_pick_index = list(zip(*union_gt))
        accuracy = F.binary_accuracy(ys[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])],
                                     ts[list(accuracy_pick_index[0]), list(accuracy_pick_index[1])])

        chainer.reporter.report({
           "accuracy": accuracy},
            self)
        return accuracy

    def loss_func(self,ys,ts):
        xp = chainer.cuda.get_array_module(ys.data)  # ys and ts shape = B x box_num x T x D
        ts = ts.reshape(-1,  ts.shape[-1])
        ys = ys.reshape(-1, ys.shape[-1])  # shape = (box_num x T) x D
        assert isinstance(ys, chainer.Variable)
        ts_data = ts.data if isinstance(ts, chainer.Variable) else ts
        cpu_gt_label = chainer.cuda.to_cpu(ts_data)  # shape = T x D
        gt_pos_index = np.nonzero(cpu_gt_label)
        cpu_pred_score = (chainer.cuda.to_cpu(ys.data) > 0).astype(np.int32)
        pred_pos_index = np.nonzero(cpu_pred_score)
        len_gt_pos = len(gt_pos_index[0]) if len(gt_pos_index[0]) > 0 else 1
        neg_pick_count = self.neg_pos_ratio * len_gt_pos
        gt_pos_index_set = set(list(zip(*gt_pos_index)))
        pred_pos_index_set = set(list(zip(*pred_pos_index)))

        false_positive_index = np.asarray(list(pred_pos_index_set - gt_pos_index_set))  # shape = n x 2
        gt_pos_index_lst = list(gt_pos_index_set)
        if neg_pick_count <= len(false_positive_index):
            choice_fp = np.random.choice(np.arange(len(false_positive_index)), size=neg_pick_count, replace=False)
            gt_pos_index_lst.extend(list(map(tuple, false_positive_index[choice_fp].tolist())))
        else:
            gt_pos_index_lst.extend(list(map(tuple, false_positive_index.tolist())))
            rest_pick_count = neg_pick_count - len(false_positive_index)
            gt_neg_index = np.where(cpu_gt_label == 0)
            gt_neg_index_set = set(list(zip(*gt_neg_index)))
            gt_neg_index_set = gt_neg_index_set - set(gt_pos_index_lst)  # remove already picked
            gt_neg_index_array = np.asarray(list(gt_neg_index_set))
            choice_rest = np.random.choice(np.arange(len(gt_neg_index_array)), size=rest_pick_count, replace=False)
            gt_pos_index_lst.extend(list(map(tuple, gt_neg_index_array[choice_rest].tolist())))

        pick_index = list(zip(*gt_pos_index_lst))
        loss = F.sigmoid_cross_entropy(ys[list(pick_index[0]), list(pick_index[1])],
                                       ts[list(pick_index[0]), list(pick_index[1])])  # 支持多label

        chainer.reporter.report({
            'loss': loss},
            self)
        return loss



