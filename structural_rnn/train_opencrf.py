import sys
sys.path.append("/home/machen/face_expr")
import argparse

import chainer


import json

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
    parser.add_argument('--pretrain', '-pr', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot', '-snap', type=int, default=100, help='snapshot iteration for save checkpoint')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--valid', '-val', default='graph_valid',
                        help='Test directory path contains test txt file')
    parser.add_argument('--test', '-tt', default='graph_test',
                        help='Test directory path contains test txt file')
    parser.add_argument('--train', '-tr', default="D:/toy/",
                        help='Train directory path contains train txt file')
    parser.add_argument('--database',default="BP4D",help="BP4D/DISFA")
    parser.add_argument('--use_pure_python',action='store_true',
                        help='you can use pure python code to check whether your optimized code works correctly')
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument("--profile","-p", action="store_true",help="whether to profile to examine speed bottleneck")


    parser.add_argument("--fold", "-fd", type=int,
                        help="which fold of K-fold")
    parser.add_argument("--split_idx", '-sp', type=int, help="which split_idx")
    parser.add_argument("--need_cache_graph", "-ng", action="store_true",
                        help="whether to cache factor graph to LRU cache")
    parser.add_argument("--eval_mode",'-eval', action="store_true", help="whether to evaluation or not")
    parser.add_argument("--proc_num","-pn", type=int, default=1)
    parser.set_defaults(test=False)
    args = parser.parse_args()
    config.OPEN_CRF_CONFIG["use_pure_python"] = args.use_pure_python
    # because we modify config.OPEN_CRF_CONFIG thus will influence the open_crf layer
    from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
    from structural_rnn.dataset.structural_RNN_dataset import S_RNNPlusDataset
    from structural_rnn.extensions.opencrf_evaluator import OpenCRFEvaluator
    from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
    from structural_rnn.updater.bptt_updater import convert
    from structural_rnn.extensions.AU_evaluator import ActionUnitEvaluator
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
    model = OpenCRFLayer(node_in_size=dataset.num_attrib_type, weight_len=crf_pact_structure.num_feature)

    if args.eval_mode:
        if not os.path.exists(args.pretrain):
            raise FileNotFoundError("pretrain file:{} not found".format(args.pretrain))
            sys.exit(-1)
        chainer.serializers.load_npz(args.pretrain, model)
        with chainer.no_backprop_mode():
            test_data = S_RNNPlusDataset(args.test,attrib_size=dataset.num_attrib_type,
                                  global_dataset=dataset,need_s_rnn=False,need_cache_factor_graph=args.need_cache_graph)
            if args.proc_num == 1:
                test_iter = chainer.iterators.SerialIterator(test_data, 1, shuffle=False)
            elif args.proc_num > 1:
                test_iter = chainer.iterators.MultiprocessIterator(test_data, batch_size=1, n_processes=args.proc_num,
                                                  repeat=False, shuffle=False, n_prefetch=10, shared_mem=31457280)
            au_evaluator = ActionUnitEvaluator(test_iter, model, device=-1,database=args.database, data_info_path=os.path.dirname(args.train) + os.sep + "data_info.json")
            observation = au_evaluator.evaluate()
            with open(args.out + os.sep + "opencrf_eval.json", "w") as file_obj:
                file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
                file_obj.flush()
        return


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
    trainer.extend(chainer.training.extensions.LogReport(trigger=print_interval,log_name="open_crf.log"))

    optimizer_snapshot_name = "{0}_{1}_{2}_crf_optimizer.npz".format(args.database, args.fold, args.split_idx)
    model_snapshot_name = "{0}_{1}_{2}_crf_model.npz".format(args.database, args.fold, args.split_idx)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'iteration'))

    trainer.extend(
            chainer.training.extensions.snapshot_object(model,
                                                        filename=model_snapshot_name),
            trigger=(args.snapshot, 'iteration'))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))
    # trainer.extend(chainer.training.extensions.snapshot(),
    #                trigger=(args.snapshot, 'epoch'))
    trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1), trigger=(10, 'epoch'))


    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss']))
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