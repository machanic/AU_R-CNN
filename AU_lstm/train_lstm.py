import chainer

import chainer.links as L

from chainer.training import extensions
from chainer.training.updater import StandardUpdater
from AU_lstm.dataset.AU_lstm_dataset import AULstmDataset
from AU_lstm.model.AU_LSTM import AU_LSTM
import argparse
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import os
import config
from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
from chainer.training.extensions import LogReport, PrintReport
from chainer import training

class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--step_size', '-ss', type=int, default=3000,
                        help='step_size for lr exponential')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=int, default=1, help='snapshot epochs for save checkpoint')
    parser.add_argument('--eval_mode', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--valid', '-v', default='',
                        help='Test directory path contains test txt file')
    parser.add_argument('--train', '-t', default="D:/toy/",
                        help='Train directory path contains train txt file')
    parser.add_argument("--database", '-d', default="BP4D")
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument("--gpu", '-g', type=int, default=0)
    parser.add_argument("--num_attrib", type=int,default=2048)
    parser.set_defaults(test=False)
    args = parser.parse_args()
    print_interval = 1, 'iteration'
    val_interval = 10, 'iteration'
    plot_interval= 100,'iteration'
    adaptive_AU_database(args.database)

    train_data = AULstmDataset(directory=args.train, database=args.database, num_attrib=args.num_attrib)
    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True)

    dataset = GlobalDataSet(num_attrib=args.num_attrib)
    au_lstm = AU_LSTM(in_size=dataset.num_attrib_type, out_size=dataset.num_label)
    model = L.Classifier(predictor=au_lstm, lossfun=au_lstm.loss_func, accfun=au_lstm.accuracy_func)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.predictor.to_gpu(device=args.gpu)


    optimizer = chainer.optimizers.RMSprop(lr=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    if args.eval_mode:
        chainer.config.train = False

    trainer.extend(PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',"main/accuracy","validation/main/loss","validation/main/accuracy"
         ]), trigger=print_interval)

    trainer.extend(LogReport(trigger=print_interval))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))
    # trainer.extend(chainer.training.extensions.snapshot(),
    #                trigger=(args.snapshot, 'epoch'))
    # trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.7), trigger=(5, 'epoch'))


    if chainer.training.extensions.PlotReport.available():
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/loss', "validation/main/loss"],
                file_name='loss_lstm.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/accuracy', "validation/main/accuracy"],
                file_name='accuracy_lstm.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
    validate_data = AULstmDataset(directory=args.valid, database=args.database,num_attrib=args.num_attrib)
    validate_iter = chainer.iterators.SerialIterator(validate_data, 1, repeat=False, shuffle=False)
    evaluator = extensions.Evaluator(
        iterator=validate_iter, target=model,  device=args.gpu)
    trainer.extend(evaluator, trigger=val_interval)

    trainer.run()

if __name__ == "__main__":
    print(chainer.__version__)
    print(chainer._cudnn_version)
    main()
