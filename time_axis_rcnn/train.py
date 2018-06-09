#!/usr/local/anaconda3/bin/python3
from __future__ import division
import sys
sys.path.insert(0, '/home/machen/face_expr')
import cProfile
import pstats
import random


from chainer.datasets import TransformDataset




from time_axis_rcnn.extensions.special_converter import concat_examples_not_string

try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

from time_axis_rcnn.constants.enum_type import OptimizerType, FasterBackboneType
from time_axis_rcnn.datasets.npz_feature_dataset import NpzFeatureDataset
from time_axis_rcnn.model.time_segment_network.faster_head_module import FasterHeadModule
from time_axis_rcnn.model.time_segment_network.faster_rcnn_backbone import FasterBackbone
from time_axis_rcnn.model.time_segment_network.tcn_backbone import TcnBackbone
from time_axis_rcnn.model.time_segment_network.faster_rcnn_train_chain import TimeSegmentRCNNTrainChain
from time_axis_rcnn.model.time_segment_network.segment_proposal_network import SegmentProposalNetwork
from time_axis_rcnn.constants.enum_type import TwoStreamMode
from time_axis_rcnn.model.time_segment_network.wrapper import Wrapper

import argparse
import os

import chainer
from chainer import training
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import config
from chainer.iterators import MultiprocessIterator, SerialIterator
from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report

class Transform(object):

    def __init__(self, mirror=True):
        self.mirror = mirror

    def __call__(self, in_data):
        # feature shape = (2048, N)
        feature, gt_segments_rgb, gt_segments_flow, seg_info, seg_labels, orig_label, _ = in_data
        if self.mirror:
            x_flip = random.choice([True, False])
            if x_flip:
                feature = feature[:, ::-1]
                W = feature.shape[1]
                x_max = W - 1 - gt_segments_rgb[:, 0]
                x_min = W - 1 - gt_segments_rgb[:, 1]
                gt_segments_rgb[:, 0] = x_min
                gt_segments_rgb[:, 1] = x_max

                W_flow = W/10.
                x_max_flow = W_flow - 1 - gt_segments_flow[:, 0]
                x_min_flow = W_flow - 1 - gt_segments_flow[:, 1]
                gt_segments_flow[:, 0] = x_min_flow
                gt_segments_flow[:, 1] = x_max_flow
                orig_label = orig_label[::-1, :]
        return feature, gt_segments_rgb, gt_segments_flow, seg_info, seg_labels, orig_label

def main():
    parser = argparse.ArgumentParser(
        description='train script of Time-axis R-CNN:')
    parser.add_argument('--pid', '-pp', default='/tmp/SpaceTime_AU_R_CNN/')
    parser.add_argument('--gpu', '-g', type=int, help='GPU ID')
    parser.add_argument('--lr', '-l', type=float, default=0.0001)
    parser.add_argument('--out', '-o', default='output_time_axis_rcnn',
                        help='Output directory')
    parser.add_argument('--database',  default='BP4D',
                        help='Output directory: BP4D/DISFA/BP4D_DISFA')
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--optimizer', type=OptimizerType,choices=list(OptimizerType))
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--roi_size', type=int, default=7)
    parser.add_argument('--snapshot', '-snap', type=int, default=5)
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument('--two_stream_mode', type=TwoStreamMode, choices=list(TwoStreamMode),
                        help='rgb_flow/ optical_flow/ rgb')
    parser.add_argument("--faster_backbone", type=FasterBackboneType,choices=list(FasterBackboneType), help='tcn/conv1d')
    parser.add_argument("--data_dir", type=str, default="/extract_features")
    parser.add_argument("--conv_layers", type=int, default=10)
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--use_paper_num_label", action="store_true", help="only to use paper reported number of labels"
                                                                           " to train")

    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    args = parser.parse_args()
    args.data_dir = config.ROOT_PATH + "/" + args.data_dir
    os.makedirs(args.pid, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    pid = str(os.getpid())
    pid_file_path = args.pid + os.path.sep + "{0}_{1}_fold_{2}.pid".format(args.database, args.fold, args.split_idx)
    with open(pid_file_path, "w") as file_obj:
        file_obj.write(pid)
        file_obj.flush()

    print('GPU: {}'.format(args.gpu))

    adaptive_AU_database(args.database)

    paper_report_label, class_num = squeeze_label_num_report(args.database, args.use_paper_num_label)
    paper_report_label_idx = list(paper_report_label.keys())

    if args.faster_backbone == FasterBackboneType.tcn:
        Bone = TcnBackbone
    elif args.faster_backbone == FasterBackboneType.conv1d:
        Bone = FasterBackbone

    if args.two_stream_mode == TwoStreamMode.rgb or args.two_stream_mode == TwoStreamMode.optical_flow:
        faster_extractor_backbone = Bone(args.conv_layers, args.feature_dim, 1024)
        faster_head_module = FasterHeadModule(args.feature_dim, class_num + 1, args.roi_size)  # note that the class number here must include background
        initialW = chainer.initializers.Normal(0.001)
        spn = SegmentProposalNetwork(1024, n_anchors=len(config.ANCHOR_SIZE), initialW=initialW)
        train_chain = TimeSegmentRCNNTrainChain(faster_extractor_backbone, faster_head_module, spn)
        model = Wrapper(train_chain, two_stream_mode=args.two_stream_mode)

    elif args.two_stream_mode == TwoStreamMode.rgb_flow:
        faster_extractor_backbone = Bone(args.conv_layers, args.feature_dim, 1024)
        faster_head_module = FasterHeadModule(args.feature_dim, class_num + 1,
                                              args.roi_size)  # note that the class number here must include background
        initialW = chainer.initializers.Normal(0.001)
        spn = SegmentProposalNetwork(1024, n_anchors=len(config.ANCHOR_SIZE), initialW=initialW)
        train_chain = TimeSegmentRCNNTrainChain(faster_extractor_backbone, faster_head_module, spn)

        # faster_extractor_backbone_flow = FasterBackbone(args.database, args.conv_layers, args.feature_dim, 1024)
        # faster_head_module_flow = FasterHeadModule(1024, class_num + 1,
        #                                       args.roi_size)  # note that the class number here must include background
        # initialW = chainer.initializers.Normal(0.001)
        # spn_flow = SegmentProposalNetwork(1024, n_anchors=len(config.ANCHOR_SIZE), initialW=initialW)
        # train_chain_flow = TimeSegmentRCNNTrainChain(faster_extractor_backbone_flow, faster_head_module_flow, spn_flow)
        # time_seg_train_chain_list = [train_chain_rgb, train_chain_flow]
        model = Wrapper(train_chain, two_stream_mode=args.two_stream_mode)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()


    optimizer = None
    if args.optimizer == OptimizerType.AdaGrad:
        optimizer = chainer.optimizers.AdaGrad(
            lr=args.lr)  # 原本为MomentumSGD(lr=args.lr, momentum=0.9) 由于loss变为nan问题，改为AdaGrad
    elif args.optimizer == OptimizerType.RMSprop:
        optimizer = chainer.optimizers.RMSprop(lr=args.lr)
    elif args.optimizer == OptimizerType.Adam:
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
    elif args.optimizer == OptimizerType.SGD:
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    elif args.optimizer == OptimizerType.AdaDelta:
        optimizer = chainer.optimizers.AdaDelta()

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    data_dir = args.data_dir + "/{0}_{1}_fold_{2}/train".format(args.database, args.fold, args.split_idx)
    dataset = NpzFeatureDataset(data_dir, args.database, two_stream_mode=args.two_stream_mode,T=10.0, use_mirror_data=True)

    dataset = TransformDataset(dataset, Transform(mirror=True))

    if args.proc_num == 1:
        train_iter = SerialIterator(dataset, args.batch_size, repeat=True, shuffle=True)
    else:
        train_iter = MultiprocessIterator(dataset,  batch_size=args.batch_size,
                                          n_processes=args.proc_num,
                                      repeat=True, shuffle=True, n_prefetch=10, shared_mem=10000000)


    # BP4D_3_fold_1_resnet101@rnn@no_temporal@use_paper_num_label@roi_align@label_dep_layer@conv_lstm@sampleframe#13_model.npz
    use_paper_classnum = "use_paper_num_label" if args.use_paper_num_label else "all_avail_label"

    model_file_name = args.out + os.path.sep + \
                             'time_axis_rcnn_{0}_{1}_fold_{2}@{3}@{4}@{5}@{6}_model.npz'.format(args.database,
                                                                                args.fold, args.split_idx,
                                                                                use_paper_classnum, args.two_stream_mode,
                                                                                            args.conv_layers, args.faster_backbone)
    print(model_file_name)
    pretrained_optimizer_file_name = args.out + os.path.sep +\
                             'time_axis_rcnn_{0}_{1}_fold_{2}@{3}@{4}@{5}@{6}_optimizer.npz'.format(args.database,
                                                                                args.fold, args.split_idx,
                                                                                 use_paper_classnum, args.two_stream_mode,
                                                                                                args.conv_layers,args.faster_backbone)
    print(pretrained_optimizer_file_name)

    if os.path.exists(pretrained_optimizer_file_name):
        print("loading optimizer snatshot:{}".format(pretrained_optimizer_file_name))
        chainer.serializers.load_npz(pretrained_optimizer_file_name, optimizer)

    if os.path.exists(model_file_name):
        print("loading pretrained snapshot:{}".format(model_file_name))
        chainer.serializers.load_npz(model_file_name, model.time_seg_train_chain)

    print("only one GPU({0}) updater".format(args.gpu))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                          converter=lambda batch, device: concat_examples_not_string(batch, device, padding=0))

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer, filename=os.path.basename(pretrained_optimizer_file_name)),
        trigger=(args.snapshot, 'epoch'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model.time_seg_train_chain,
                                                    filename=os.path.basename(model_file_name)),
        trigger=(args.snapshot, 'epoch'))

    log_interval = 100, 'iteration'
    print_interval = 100, 'iteration'
    plot_interval = 100, 'iteration'
    if args.optimizer != "Adam" and args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1),
                       trigger=(20, 'epoch'))
    elif args.optimizer == "Adam":
        trainer.extend(chainer.training.extensions.ExponentialShift("alpha", 0.1, optimizer=optimizer), trigger=(10, 'epoch'))
    if args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_interval,log_name="log_{0}_{1}_{2}_fold_{3}_{4}.log".format(args.faster_backbone,
                                                                                        args.database, args.fold, args.split_idx,
                                                                                        use_paper_classnum)))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss','main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'main/accuracy',
         'main/rpn_accuracy',
         ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=100))

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/loss'],
                file_name='loss_{0}_{1}_fold_{2}_{3}.png'.format(args.database, args.fold, args.split_idx,
                                                        use_paper_classnum), trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/accuracy'],
                file_name='accuracy_{0}_{1}_fold_{2}_{3}.png'.format(args.database, args.fold, args.split_idx,
                                                        use_paper_classnum), trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.run()




if __name__ == '__main__':
    main()
