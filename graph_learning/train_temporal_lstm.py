import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from graph_learning.dataset.graph_dataset_reader import GlobalDataSet
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from graph_learning.dataset.graph_dataset import GraphDataset
from graph_learning.model.temporal_lstm.temporal_lstm import TemporalLSTM
from graph_learning.updater.bptt_updater import BPTTUpdater
import os
import matplotlib
matplotlib.use('Agg')
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=int, default=100,
                        help='snapshot epochs for save checkpoint')
    parser.add_argument('--valid', '-v', default='',
                        help='validate directory path contains validate txt file')
    parser.add_argument('--train', '-t', default="train",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',  default="BP4D",
                        help='database to train for')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help="hidden_size orignally used in open_crf")
    parser.add_argument('--eval_mode', action='store_true', help='whether to evaluate the model')
    parser.add_argument("--need_cache_graph", "-ng", action="store_true",
                        help="whether to cache factor graph to LRU cache")
    parser.add_argument("--bi_lstm", '-bilstm', action='store_true', help="Use bi_lstm as basic component of temporal_lstm")
    parser.add_argument("--num_attrib",type=int, default=2048, help="node feature dimension")
    parser.add_argument("--resume", action="store_true", help="whether to load npz pretrained file")
    parser.add_argument("--snap_individual", action="store_true",
                        help="whether to snapshot each individual epoch/iteration")

    parser.set_defaults(test=False)
    args = parser.parse_args()
    print_interval = 1, 'iteration'
    val_interval = 5, 'iteration'

    adaptive_AU_database(args.database)


    # for the StructuralRNN constuctor need first frame factor graph_backup
    dataset = GlobalDataSet(num_attrib=args.num_attrib)
    model = TemporalLSTM(box_num=config.BOX_NUM[args.database], in_size=args.num_attrib,
                         out_size=dataset.label_bin_len,
                         use_bi_lstm=args.bi_lstm, initialW=None)

    train_data = GraphDataset(args.train, attrib_size=args.hidden_size, global_dataset=dataset, need_s_rnn=True,
                              need_cache_factor_graph=args.need_cache_graph)

    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True, repeat=True)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    updater = BPTTUpdater(train_iter, optimizer, int(args.gpu))
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    print_interval = (1, 'iteration')

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', "main/accuracy"
         ]), trigger=print_interval)
    log_name = "temporal_lstm.log"
    trainer.extend(chainer.training.extensions.LogReport(trigger=print_interval,log_name=log_name))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1, training_length=(args.epoch, 'epoch')))
    optimizer_snapshot_name = "{0}_temporal_lstm_optimizer.npz".format(args.database)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'iteration'))

    if not args.snap_individual:
        model_snapshot_name = "{0}_temporal_lstm_model.npz".format(args.database)
        trainer.extend(
            chainer.training.extensions.snapshot_object(model, filename=model_snapshot_name),
            trigger=(args.snapshot, 'iteration'))
    else:
        model_snapshot_name = "{0}_temporal_lstm_model_".format(args.database) + "{.updater.iteration}.npz"
        trainer.extend(chainer.training.extensions.snapshot_object(model, filename=model_snapshot_name),
                       trigger=(args.snapshot, 'iteration'))

    trainer.extend(chainer.training.extensions.ExponentialShift('lr',0.7), trigger=(5, "epoch"))

    # load pretrained file
    if not args.snap_individual:
        if args.resume and os.path.exists(args.out + os.sep + model_snapshot_name):
            print("loading model_snapshot_name to model")
            chainer.serializers.load_npz(args.out + os.sep + model_snapshot_name, model)
    else:
        if args.resume:
            file_lst = [filename[filename.rindex("_")+1:filename.rindex(".")] for filename in os.listdir(args.out)]
            file_no = sorted(map(int, file_lst))[-1]
            model_snapshot_name = "{0}_temporal_lstm_model_{1}.npz".format(args.database, file_no)
            chainer.serializers.load_npz(args.out+ os.sep + model_snapshot_name, model)

    if args.resume and os.path.exists(args.out + os.sep + optimizer_snapshot_name):
        print("loading optimizer_snapshot_name to optimizer")
        chainer.serializers.load_npz(args.out + os.sep + optimizer_snapshot_name, optimizer)


    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss'],
                                                              file_name="train_loss.png"),
                                                              trigger=(100,"iteration"))
        # trainer.extend(chainer.training.extensions.PlotReport(['opencrf_val/F1','opencrf_val/accuracy'],
        #                                                       file_name="val_f1.png"), trigger=val_interval)



    # if args.valid:
    #     valid_data = S_RNNPlusDataset(args.valid, attrib_size=args.hidden_size, global_dataset=dataset,
    #                                   need_s_rnn=True,need_cache_factor_graph=args.need_cache_graph)  # attrib_size控制open-crf层的weight长度
    #     validate_iter = chainer.iterators.SerialIterator(valid_data, 1, shuffle=False, repeat=False)
    #     crf_evaluator = OpenCRFEvaluator(iterator=validate_iter, target=model, device=args.gpu)
    #     trainer.extend(crf_evaluator, trigger=val_interval, name="opencrf_val")

    trainer.run()


if __name__ == "__main__":
    main()