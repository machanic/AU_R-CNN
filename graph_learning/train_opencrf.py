import sys
sys.path.append("/home/machen/face_expr")
import argparse

import chainer


from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

from chainer.training import StandardUpdater
from chainer.training.extensions.print_report import PrintReport


import os
import config
import cProfile
import pstats
import matplotlib
matplotlib.use('Agg')



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
    parser.add_argument('--pretrain', '-pr', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot', '-snap', type=int, default=100, help='snapshot iteration for save checkpoint')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--valid', '-val', default='',
                        help='Test directory path contains test txt file')
    parser.add_argument('--test', '-tt', default='graph_test',
                        help='Test directory path contains test txt file')
    parser.add_argument('--train', '-tr', default="D:/toy/",
                        help='Train directory path contains train txt file')
    parser.add_argument('--train_edge', default="all", help="train temporal/all to comparision")
    parser.add_argument('--database',default="BP4D",help="BP4D/DISFA")
    parser.add_argument('--use_pure_python',action='store_true',
                        help='you can use pure python code to check whether your optimized code works correctly')
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument("--profile","-p", action="store_true",help="whether to profile to examine speed bottleneck")
    parser.add_argument("--num_attrib", type=int, default=2048, help="node feature dimension")
    parser.add_argument("--need_cache_graph", "-ng", action="store_true",
                        help="whether to cache factor graph to LRU cache")
    parser.add_argument("--eval_mode",'-eval', action="store_true", help="whether to evaluation or not")
    parser.add_argument("--proc_num","-pn", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="resume from pretrained model")
    parser.set_defaults(test=False)
    args = parser.parse_args()
    config.OPEN_CRF_CONFIG["use_pure_python"] = args.use_pure_python
    # because we modify config.OPEN_CRF_CONFIG thus will influence the open_crf layer
    from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
    from graph_learning.dataset.structural_RNN_dataset import S_RNNPlusDataset
    from graph_learning.extensions.opencrf_evaluator import OpenCRFEvaluator
    from graph_learning.dataset.graph_dataset_reader import GlobalDataSet
    from graph_learning.updater.bptt_updater import convert
    from graph_learning.extensions.AU_evaluator import ActionUnitEvaluator
    if args.use_pure_python:

        from graph_learning.model.open_crf.pure_python.open_crf_layer import OpenCRFLayer
    else:
        from graph_learning.model.open_crf.cython.open_crf_layer import OpenCRFLayer

    print_interval = 1, 'iteration'
    val_interval = (5, 'iteration')
    adaptive_AU_database(args.database)
    root_dir = os.path.dirname(os.path.dirname(args.train))
    dataset = GlobalDataSet(num_attrib=args.num_attrib, train_edge=args.train_edge)
    file_name = list(filter(lambda e:e.endswith(".txt"),os.listdir(args.train)))[0]
    sample = dataset.load_data(args.train + os.sep + file_name)
    print("pre load done")

    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=dataset.num_attrib_type, need_s_rnn=False)
    model = OpenCRFLayer(node_in_size=dataset.num_attrib_type, weight_len=crf_pact_structure.num_feature)

    train_str = args.train
    if train_str[-1] == "/":
        train_str = train_str[:-1]
    trainer_keyword = os.path.basename(train_str)
    trainer_keyword_tuple = tuple(trainer_keyword.split("_"))
    LABEL_SPLIT = config.BP4D_LABEL_SPLIT if args.database == "BP4D" else config.DISFA_LABEL_SPLIT
    if trainer_keyword_tuple not in LABEL_SPLIT:
        return
    # assert "_" in trainer_keyword



    train_data = S_RNNPlusDataset(args.train, attrib_size=dataset.num_attrib_type,
                                  global_dataset=dataset,need_s_rnn=False,need_cache_factor_graph=args.need_cache_graph)
    if args.proc_num == 1:
        train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=True)
    elif args.proc_num > 1:
        train_iter = chainer.iterators.MultiprocessIterator(train_data, batch_size=1, n_processes=args.proc_num,
                                                           repeat=True, shuffle=True, n_prefetch=10,
                                                           shared_mem=31457280)
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
         'opencrf_val/main/accuracy',
         ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.LogReport(trigger=print_interval,log_name="open_crf_{}.log".format(trainer_keyword)))

    optimizer_snapshot_name = "{0}_{1}_opencrf_optimizer.npz".format(trainer_keyword, args.database)
    model_snapshot_name = "{0}_{1}_opencrf_model.npz".format(trainer_keyword, args.database)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'iteration'))

    trainer.extend(
            chainer.training.extensions.snapshot_object(model,
                                                        filename=model_snapshot_name),
            trigger=(args.snapshot, 'iteration'))

    if args.resume and os.path.exists(args.out + os.sep + model_snapshot_name):
        print("loading model_snapshot_name to model")
        chainer.serializers.load_npz(args.out + os.sep + model_snapshot_name, model)
    if args.resume and os.path.exists(args.out + os.sep + optimizer_snapshot_name):
        print("loading optimizer_snapshot_name to optimizer")
        chainer.serializers.load_npz(args.out + os.sep + optimizer_snapshot_name, optimizer)


    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))
    # trainer.extend(chainer.training.extensions.snapshot(),
    #                trigger=(args.snapshot, 'epoch'))

    # trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.9), trigger=(1, 'epoch'))


    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss'],
                                                              file_name="{}_train_loss.png".format(trainer_keyword)),
                                                              trigger=(100,"iteration"))
        trainer.extend(chainer.training.extensions.PlotReport(['opencrf_val/F1','opencrf_val/accuracy'],
                                                              file_name="{}_val_f1.png".format(trainer_keyword)), trigger=val_interval)

    if args.valid:
        valid_data = S_RNNPlusDataset(args.valid, attrib_size=dataset.num_attrib_type,
                                      global_dataset=dataset, need_s_rnn=False, need_cache_factor_graph=args.need_cache_graph)
        validate_iter = chainer.iterators.SerialIterator(valid_data, 1, repeat=False, shuffle=False)
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