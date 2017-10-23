import chainer
from chainer.datasets import get_mnist as get_mnist
import chainer.functions as F
import chainer.links as L
from chainer.dataset import concat_examples
import numpy as np


class Model(chainer.Chain):

    def __init__(self, in_size, out_size):
        super(Model, self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1, in_size, out_size, 0.5)

    def __call__(self, xs):
        xs = [x for x in xs]
        _, _, y_lst = self.lstm(None, None, xs)
        # final_y_lst = F.stack([ys[-1] for ys in y_lst], axis=0) # must retain chainer.Variable
        final_y_lst = F.stack(y_lst)  # shape = B x T x D
        return final_y_ls.t

train, test = get_mnist(ndim=2)

from chainer import training


class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()

        # Concatenate the word IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        xs, ts = self.converter(batch, self.device)
        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(chainer.Variable(xs), chainer.Variable(ts))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

def accuracy(ys, ts):


    yy = ys[:, ys.shape[1]//2:, :] # B x T/2 x D
    ts = F.tile(ts, reps=(yy.shape[1], 1)).transpose() # B, => T/2 x B => B x T/2
    accu = F.accuracy(yy.reshape(-1,10), ts.reshape(-1))
    return accu

def sum_sigmoid_cross_entropy(ys, ts):
    xp = chainer.cuda.get_array_module(ts.data)
    ts = ts.data
    t_bin = xp.zeros((len(ts), 10), dtype=xp.int32) # shape = B x 10
    # FIXME 做一个修改，不从最后一个时间步去获取loss，而是从中间的时间步开始的各个时间步的loss累加起来，用F.sigmoid_cross_entropy自动做
    t_bin[np.arange(len(ts)), ts] = 1
    cpu_gt_roi_label_lst = chainer.cuda.to_cpu(t_bin)
    pos_index = np.nonzero(cpu_gt_roi_label_lst)
    neg_index = list(np.where(cpu_gt_roi_label_lst == 0))
    neg_length = 3 * len(pos_index[0])
    pick_neg = np.random.choice(range(len(neg_index[0])), size=neg_length, replace=False)
    neg_index[0] = neg_index[0][pick_neg]
    neg_index[1] = neg_index[1][pick_neg]
    row_index = np.concatenate((pos_index[0], neg_index[0]), axis=0)
    col_index = np.concatenate((pos_index[1], neg_index[1]), axis=0)
    # assert gt_roi_label_lst[pos_index[0], pos_index[1]].any()
    # assert not gt_roi_label_lst[neg_index[0], neg_index[1]].any()
    ys = F.transpose(ys, (1, 0, 2)) # shape T x B x D
    t_bin = F.tile(t_bin, reps=(len(ys), 1, 1))  # shape = B x 10 => T x B x 10
    ys = ys[len(ys)//2:, row_index, col_index]
    t_bin = t_bin[len(t_bin)//2:, row_index, col_index]
    loss = F.sigmoid_cross_entropy(ys, t_bin)
    return loss

def convert(converter, batch, device):
    xs, ts =converter(batch,device=device)
    return chainer.Variable(xs), chainer.Variable(ts)

import time
from chainer import training
from chainer.training import extensions

def train_and_test(enable_cupy, n_epoch):
    training_start = time.clock()
    log_trigger = 100, 'iteration'
    model = Model(in_size=28,out_size=10)
    classifier_model = L.Classifier(model, lossfun=sum_sigmoid_cross_entropy, accfun=accuracy)
    device = -1
    if enable_cupy:
        model.to_gpu()
        chainer.cuda.get_device(0).use()
        device = 0
    batchsize = 100
    optimizer = chainer.optimizers.RMSprop(lr=0.001)
    optimizer.setup(classifier_model)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
    updater = BPTTUpdater(train_iter, optimizer, device)
    # updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='out')
    trainer.extend(extensions.dump_graph('main/loss'))
    valid_trigger = 10000, 'iteration'
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=100))
    trainer.extend(extensions.Evaluator(test_iter, classifier_model,
                                        converter=lambda batch, device: convert(concat_examples, batch, device),
                                        device=device), trigger=valid_trigger)
    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']), trigger=log_trigger)
    trainer.run()
    elapsed_time = time.clock() - training_start
    print('Elapsed time: %3.3f' % elapsed_time)

train_and_test(True, n_epoch=100)
