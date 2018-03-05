import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from chainer.iterators import MultiprocessIterator, SerialIterator

from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from simple_graph_learning.model.space_time_net.space_time_rnn import SpaceTimeRNN
from simple_graph_learning.dataset.simple_feature_dataset import SimpleFeatureDataset
from simple_graph_learning.model.space_time_net.enum_type import SpatialEdgeMode, RecurrentType
from simple_graph_learning.iterators.batch_keep_order_iterator import BatchKeepOrderIterator
from collections import OrderedDict
import os
import matplotlib
matplotlib.use('Agg')
import config
import re
from chainer.dataset import concat_examples


def parse_fold_split_idx(train_dir):
    pattern = re.compile(".*/(.*?)_(\d)_fold_(\d)/.*", re.DOTALL)
    matcher = pattern.match(train_dir)
    return_dict = {}
    if matcher:
        database = matcher.group(1)
        fold = matcher.group(2)
        split_idx = matcher.group(3)
        return_dict["database"] = database
        return_dict["fold"] = fold
        return_dict["split_idx"] = split_idx
    return return_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=25,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument('--step_size', '-ss', type=int, default=10,
                        help='step_size for lr exponential')
    parser.add_argument('--proc', '-pc', type=int, default=1,
                        help='proc_num for multi-process fetch data')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='r_srnn_result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', '-snap', type=int, default=1, help='snapshot epochs for save checkpoint')
    parser.add_argument('--train', '-t', default=" /home/machen/dataset/new_graph/BP4D_3_fold_1/train",
                        help='the folder path that contains train npz file')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of different video clip')
    parser.add_argument('--previous_frame','-prev', type=int, default=60, help='previous frame before random sample')
    parser.add_argument('--lr', '-l', type=float, default=0.01)
    parser.add_argument('--spatial_edge_mode', type=SpatialEdgeMode, choices=list(SpatialEdgeMode), help='1:all_edge, 2:configure_edge, 3:no_edge')
    parser.add_argument('--temporal_edge_mode',type=RecurrentType, choices=list(RecurrentType), help='1:rnn, 2:attention_block, 3.point-wise feed forward(no temporal)')
    parser.add_argument("--num_attrib", type=int, default=2048, help="number of dimension of each node feature")
    parser.add_argument('--attn_heads', type=int, default=16, help='attention heads number')
    parser.add_argument('--layers', type=int, default=1, help='edge rnn and node rnn layer')
    parser.add_argument("--use_paper_num_label", action="store_true", help="only to use paper reported number of labels"
                                                                           " to train")
    parser.add_argument("--bi_lstm", action="store_true", help="whether to use bi-lstm as Edge/Node RNN")
    parser.add_argument('--weight_decay',type=float,default=0.0005, help="weight decay")
    parser.add_argument("--proc_num",'-proc', type=int,default=1, help="process number of dataset reader")
    parser.add_argument("--file_pic_num", '-pic_num', type=int, default=200, help="process number of dataset reader")
    parser.add_argument('--resume_model', '-model', help='The relative path to restore model file')
    parser.add_argument("--snap_individual", action="store_true", help='whether to snap shot each fixed step into '
                                                                       'individual model file')
    parser.add_argument("--sample_frame", type=int, default=25, help='LSTM previous frame sample number')

    parser.set_defaults(test=False)
    args = parser.parse_args()

    parse_dict = parse_fold_split_idx(args.train)
    database = parse_dict['database']

    print_interval = 1, 'iteration'
    val_interval = 1000, "iteration"
    print("""
    gpu : {0}
    ======================================
        argument: 
            spatial_edge_mode:{1}
            temporal_edge_mode:{2}
    ======================================
    """.format(args.gpu, args.spatial_edge_mode, args.temporal_edge_mode))
    adaptive_AU_database(database)

    paper_report_label = OrderedDict()
    if args.use_paper_num_label:
        for AU_idx,AU in sorted(config.AU_SQUEEZE.items(), key=lambda e:int(e[0])):
            if database == "BP4D":
                paper_use_AU = config.paper_use_BP4D
            elif database =="DISFA":
                paper_use_AU = config.paper_use_DISFA
            if AU in paper_use_AU:
                paper_report_label[AU_idx] = AU
    paper_report_label_idx = list(paper_report_label.keys())
    if not paper_report_label_idx:
        paper_report_label_idx = None
        class_num = len(config.AU_SQUEEZE)
    else:
        class_num = len(paper_report_label_idx)
    model = SpaceTimeRNN(database, args.layers, args.num_attrib, class_num, None, spatial_edge_model=args.spatial_edge_mode,
                           recurrent_block_type=args.temporal_edge_mode, attn_heads=args.attn_heads,
                           bi_lstm=args.bi_lstm)

    train_data = SimpleFeatureDataset(args.train, database, args.file_pic_num, args.previous_frame, args.sample_frame,
                                      paper_report_label_idx)
    if args.proc <= 1:
        train_iter = SerialIterator(train_data, args.batch_size, repeat=True, shuffle=True)
    else:
        train_iter = MultiprocessIterator(train_data,args.batch_size, shuffle=True, repeat=True,
                                          n_processes=args.proc, n_prefetch=10, shared_mem=10000000)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)


    specific_key = "all_AU_train"
    if paper_report_label_idx:
        specific_key = "paper_AU_num_train"

    optimizer_snapshot_name = "r_srnn_optimizer@{0}@{1}@{2}@{3}.npz".format(database,specific_key,
                                                                          args.spatial_edge_mode,
                                                                          args.temporal_edge_mode)
    model_snapshot_name = "r_srnn_model@{0}@{1}@{2}@{3}.npz".format(database, specific_key,
                                                                                      args.spatial_edge_mode,
                                                                                      args.temporal_edge_mode)
    if args.snap_individual:
        model_snapshot_name = "{0}@{1}@r_srnn_model".format(database,specific_key)

        model_snapshot_name += "@{0}@{1}_".format(args.spatial_edge_mode,
                                                             args.temporal_edge_mode)
        model_snapshot_name += "{.updater.iteration}.npz"
    if os.path.exists(args.out + os.sep + model_snapshot_name):
        print("found trained model file. load trained file: {}".format(args.out + os.sep + model_snapshot_name))
        chainer.serializers.load_npz(args.out + os.sep + model_snapshot_name, model)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=int(args.gpu),
                                               converter=lambda batch, device: concat_examples(batch, device, padding=0))
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = (1, 'iteration')

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=print_interval)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', "main/accuracy",
         ]), trigger=print_interval)

    log_name = "spacetime_rnn_{0}@{1}@{2}.log".format(database,
                                                                      args.spatial_edge_mode,
                                                                      args.temporal_edge_mode)
    trainer.extend(chainer.training.extensions.LogReport(trigger=interval,log_name=log_name))
    # trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1, training_length=(args.epoch, 'epoch')))

    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=optimizer_snapshot_name),
        trigger=(args.snapshot, 'epoch'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model,
                                                    filename=model_snapshot_name),
        trigger=(args.snapshot, 'epoch'))

    trainer.extend(chainer.training.extensions.ExponentialShift('lr',0.1), trigger=(10, "epoch"))

    if args.resume_model and os.path.exists(args.resume_model):
        print("loading model_snapshot_name to model")
        chainer.serializers.load_npz(args.resume_model, model)
    if args.resume_model and os.path.exists(optimizer_snapshot_name):
        print("loading optimizer_snapshot_name to optimizer")
        chainer.serializers.load_npz(optimizer_snapshot_name, optimizer)

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(chainer.training.extensions.PlotReport(['main/loss'],
                                                              file_name="train_loss.png"),
                                                              trigger=val_interval)
        trainer.extend(chainer.training.extensions.PlotReport(['main/accuracy'],
                                                              file_name="train_accuracy.png"), trigger=val_interval)

    trainer.run()

    # cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()

    # # evaluate the final model
    # test_data = S_RNNPlusDataset(args.valid, attrib_size=args.hidden_size, global_dataset=dataset)
    # test_iter = chainer.iterators.SerialIterator(test_data, 1, shuffle=False, repeat=False)
    # gpu = int(args.gpu)
    # chainer.cuda.get_device_from_id(gpu).use()
    # if args.with_crf:
    #     evaluator = OpenCRFEvaluator(iterator=test_iter, target=model, device=-1)
    # else:
    #     evaluator = ActionUnitEvaluator(iterator=validate_iter, target=model, device=-1)
    # result = evaluator()
    # print("final test data result: loss: {0}, accuracy:{1}".format(result["main/loss"], result["main/accuracy"]))

if __name__ == "__main__":
    main()