import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from structural_rnn.dataset.structural_RNN_dataset import S_RNNPlusDataset
from structural_rnn.extensions.opencrf_evaluator import OpenCRFEvaluator
from structural_rnn.model.temporal_lstm.temporal_lstm_plus_crf import TemporalLSTMPlus
from structural_rnn.updater.bptt_updater import BPTTUpdater
from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
from structural_rnn.updater.bptt_updater import convert
import os
from structural_rnn.trigger.EarlyStopTrigger import EarlyStoppingTrigger
import matplotlib
matplotlib.use('Agg')
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=int, default=100, help='snapshot epochs for save checkpoint')
    parser.add_argument('--valid', '-v', default='',
                        help='validate directory path contains validate txt file')
    parser.add_argument('--train', '-t', default="train",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',  default="BP4D",
                        help='database to train for')
    parser.add_argument("--stop_eps", '-eps', type=float, default=1e-4, help="f - old_f < eps ==> early stop")
    parser.add_argument('--with_crf', '-crf', action='store_true', help='whether to use open crf layer')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--crf_lr', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=1024, help="if you want to use open-crf layer, this hidden_size is node dimension input of open-crf")
    parser.add_argument('--eval_mode', action='store_true', help='whether to evaluate the model')

    parser.add_argument("--need_cache_graph", "-ng", action="store_true",
                        help="whether to cache factor graph to LRU cache")
    parser.add_argument("--bi_lstm", '-bilstm', action='store_true', help="Use bi_lstm as basic component of S-RNN")
    parser.add_argument("--num_attrib",type=int, default=2048, help="node feature dimension")
    parser.add_argument("--resume", action="store_true", help="whether to load npz pretrained file")
    parser.set_defaults(test=False)
    args = parser.parse_args()
    print_interval = 1, 'iteration'
    val_interval = 5, 'iteration'

    adaptive_AU_database(args.database)
    train_str = args.train
    if train_str[-1] == "/":
        train_str = train_str[:-1]
    trainer_keyword = os.path.basename(train_str)
    assert "_" in trainer_keyword

    # for the StructuralRNN constuctor need first frame factor graph_backup
    dataset = GlobalDataSet(num_attrib=args.num_attrib)
    file_name = list(filter(lambda e: e.endswith(".txt"), os.listdir(args.train)))[0]
    sample = dataset.load_data(args.train + os.sep + file_name)  # we load first sample for construct S-RNN, it must passed to constructor argument
    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=args.hidden_size)  # 只读取其中的一个视频的第一帧，由于node个数相对稳定，因此可以construct RNN

    model = TemporalLSTMPlus(crf_pact_structure, sequence_num=config.BOX_NUM[args.database],
                             in_size=dataset.num_attrib_type, out_size=dataset.num_label,
                              hidden_size=args.hidden_size, with_crf=args.with_crf, use_bi_lstm=args.bi_lstm)

    # note that the following code attrib_size will be used by open_crf for parameter number, thus we cannot pass dataset.num_attrib_type!
    train_data = S_RNNPlusDataset(args.train,  attrib_size=args.hidden_size, global_dataset=dataset, need_s_rnn=True,
                                  need_cache_factor_graph=args.need_cache_graph)

    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True, repeat=True)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.temporal_lstm.to_gpu(args.gpu)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    updater = BPTTUpdater(train_iter, optimizer, int(args.gpu))
    early_stop = EarlyStoppingTrigger(args.epoch, key='main/loss', eps=float(args.stop_eps))
    if args.with_crf:
        trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, "epoch"), out=args.out)
        model.open_crf.W.update_rule.hyperparam.lr = float(args.crf_lr)
        model.open_crf.to_cpu()
    else:
        trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    print_interval = (1, 'iteration')

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', "main/accuracy","opencrf_val/main/hit",  # "opencrf_validation/main/U_hit",
         "opencrf_val/main/miss",  # "opencrf_validation/main/U_miss",
         "opencrf_val/main/F1",  # "opencrf_validation/main/U_F1"
         'opencrf_val/main/accuracy',
         ]), trigger=print_interval)
    log_name = "temporal_lstm_{}.log".format(trainer_keyword)
    trainer.extend(chainer.training.extensions.LogReport(trigger=print_interval,log_name=log_name))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1, training_length=(args.epoch, 'epoch')))
    optimizer_snapshot_name = "{0}_{1}_temporal_lstm_optimizer.npz".format(trainer_keyword, args.database)
    model_snapshot_name = "{0}_{1}_temporal_lstm{2}_model.npz".format(trainer_keyword, args.database,
                                                                           "_crf" if args.with_crf else "")
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'iteration'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model,
                                                    filename=model_snapshot_name),
        trigger=(args.snapshot, 'iteration'))
    trainer.extend(chainer.training.extensions.ExponentialShift('lr',0.7), trigger=(5, "epoch"))

    if args.resume and os.path.exists(args.out + os.sep + model_snapshot_name):
        print("loading model_snapshot_name to model")
        chainer.serializers.load_npz(args.out + os.sep + model_snapshot_name, model)
    if args.resume and os.path.exists(args.out + os.sep + optimizer_snapshot_name):
        print("loading optimizer_snapshot_name to optimizer")
        chainer.serializers.load_npz(args.out + os.sep + optimizer_snapshot_name, optimizer)


    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss'],
                                                              file_name="{}_train_loss.png".format(trainer_keyword)),
                                                              trigger=(100,"iteration"))
        trainer.extend(chainer.training.extensions.PlotReport(['opencrf_val/F1','opencrf_val/accuracy'],
                                                              file_name="{}_val_f1.png".format(trainer_keyword)), trigger=val_interval)



    if args.valid:
        valid_data = S_RNNPlusDataset(args.valid, attrib_size=args.hidden_size, global_dataset=dataset,
                                      need_s_rnn=True,need_cache_factor_graph=args.need_cache_graph)  # attrib_size控制open-crf层的weight长度
        validate_iter = chainer.iterators.SerialIterator(valid_data, 1, shuffle=False, repeat=False)
        crf_evaluator = OpenCRFEvaluator(iterator=validate_iter, target=model, device=args.gpu)
        trainer.extend(crf_evaluator, trigger=val_interval, name="opencrf_val")

    trainer.run()


if __name__ == "__main__":
    main()