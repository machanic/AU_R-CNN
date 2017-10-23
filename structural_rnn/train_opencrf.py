
import argparse

import chainer




from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

from chainer.training import StandardUpdater
from chainer.training.extensions.print_report import PrintReport


import os
import config
import cProfile
import pstats


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
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot', '-snap', type=float, default=1, help='snapshot epochs for save checkpoint')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--valid', '-v', default='graph_valid',
                        help='Test directory path contains test txt file')
    parser.add_argument('--train', '-t', default="D:/toy/",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',default="BP4D",help="BP4D/DISFA")
    parser.add_argument('--use_pure_python',action='store_true',
                        help='you can use pure python code to check whether your optimized code works correctly')
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument("--profile","-p", action="store_true",help="whether to profile to examine speed bottleneck")
    parser.set_defaults(test=False)
    args = parser.parse_args()
    config.OPEN_CRF_CONFIG["use_pure_python"] = args.use_pure_python

    from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
    from structural_rnn.dataset.structural_RNN_dataset import S_RNNPlusDataset
    from structural_rnn.extensions.opencrf_evaluator import OpenCRFEvaluator
    from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
    from structural_rnn.updater.bptt_updater import convert
    if args.use_pure_python:

        from structural_rnn.model.open_crf.pure_python.open_crf_layer import OpenCRFLayer
    else:
        from structural_rnn.model.open_crf.cython.open_crf_layer import OpenCRFLayer

    print_interval = 1, 'iteration'
    val_interval = (1, 'iteration')
    adaptive_AU_database(args.database)
    dataset = GlobalDataSet(os.path.dirname(args.train) + os.sep + "data_info.json")
    file_name = os.listdir(args.train)[0]
    sample = dataset.load_data(args.train + os.sep + file_name)
    print("pre load done")

    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=dataset.num_attrib_type, need_s_rnn=False)
    train_data = S_RNNPlusDataset(args.train, attrib_size=dataset.num_attrib_type,
                                  global_dataset=dataset,need_s_rnn=False)
    # valid_data = OpenCRFDataset(args.valid)
    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=False)
    validate_iter = chainer.iterators.SerialIterator(train_data, 1, repeat=False, shuffle=False)


    model = OpenCRFLayer(node_in_size=dataset.num_attrib_type, weight_len=crf_pact_structure.num_feature)

    optimizer = chainer.optimizers.SGD(lr=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    updater = StandardUpdater(train_iter, optimizer, converter=convert)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = 1
    if args.test_mode:
        chainer.config.train = False

    trainer.extend(PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',"opencrf_val/main/hit",#"opencrf_validation/main/U_hit",
         "opencrf_val/main/miss",#"opencrf_validation/main/U_miss",
         "opencrf_val/main/F1",#"opencrf_validation/main/U_F1"
         'opencrf_val/main/accuracy'
         ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.LogReport(trigger=print_interval))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))
    # trainer.extend(chainer.training.extensions.snapshot(),
    #                trigger=(args.snapshot, 'epoch'))
    # trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.95), trigger=(1, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss']))

    evaluator = OpenCRFEvaluator(
        iterator=validate_iter, target=model,  device=-1)
    trainer.extend(evaluator, trigger=val_interval)

    if args.profile:
        cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        trainer.run()


if __name__ == "__main__":
    main()