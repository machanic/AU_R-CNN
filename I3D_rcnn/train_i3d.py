#!/usr/local/anaconda3/bin/python3
from __future__ import division

import sys

from I3D_rcnn.I3D.i3d import I3DFeatureExtractor, I3DRoIHead

sys.path.insert(0, '/home/machen/face_expr')


try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

import argparse
import numpy as np
import os

import chainer
from chainer import training

from chainer.datasets import TransformDataset
from I3D_rcnn.I3D.au_rcnn_train_chain import AU_RCNN_ROI_Extractor, AU_RCNN_TrainChainLoss
from I3D_rcnn.I3D.wrapper import Wrapper
from I3D_rcnn import transforms
from I3D_rcnn.datasets.AU_video_dataset import AU_video_dataset
from I3D_rcnn.datasets.AU_dataset import AUDataset
from I3D_rcnn.constants.enum_type import TwoStreamMode
from chainer.dataset import concat_examples
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import config
from chainer.iterators import MultiprocessIterator, SerialIterator
from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report


# new feature support:
# 1. 支持resnet101/resnet50/VGG的模块切换; 3.支持多GPU切换
# 5.支持是否进行validate（每制定epoch的时候）
# 6. 支持读取pretrained model从vgg_face或者imagenet的weight 7. 支持优化算法的切换，比如AdaGrad或RMSprop
# 8. 使用memcached

class Transform3D(object):

    def __init__(self, au_rcnn, mirror=False):
        self.au_rcnn = au_rcnn
        self.mirror = mirror

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.au_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        assert len(np.where(bbox < 0)[0]) == 0
        # horizontally flip and random shift box
        if self.mirror:
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label


class SubStractMean(object):
    def __init__(self, mean_file):
        self.mean = np.load(mean_file)

    def __call__(self, img):
        _, H, W = img.shape
        img = (img - self.mean).astype(np.float32, copy=False)
        return img

def filter_last_checkpoint_filename(file_name_list, file_type, key_str):
    last_snap_epoch = 0
    ret_name = ""
    for file_name in file_name_list:
        if file_type in file_name and key_str in file_name and "snapshot_" in file_name:
            snapshot = file_name[file_name.index("snapshot_")+len("snapshot_"):file_name.rindex(".")]
            if not snapshot.isdigit():
                continue
            snapshot = int(snapshot)
            if last_snap_epoch < snapshot:
                last_snap_epoch = snapshot
                ret_name = file_name
    return ret_name



def main():
    parser = argparse.ArgumentParser(
        description='I3D R-CNN train:')
    parser.add_argument('--pid', '-pp', default='/tmp/SpaceTime_AU_R_CNN/')
    parser.add_argument('--gpu', '-g', nargs='+', type=int, help='GPU ID, multiple GPU split by space')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--out', '-o', default='i3d_result',
                        help='Output directory')
    parser.add_argument('--database',  default='BP4D',
                        help='Output directory: BP4D/DISFA/BP4D_DISFA')
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--snapshot', '-snap', type=int, default=1000)
    parser.add_argument('--mean', default=config.ROOT_PATH+"BP4D/idx/mean_no_enhance.npy", help='image mean .npy file')
    parser.add_argument('--backbone', default="mobilenet_v1", help="vgg/resnet101/mobilenet_v1 for train")
    parser.add_argument('--optimizer', default='SGD', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
    parser.add_argument('--pretrained_rgb', help='imagenet/mobilenet_v1/resnet101/*.npz')
    parser.add_argument('--pretrained_flow', help="path of optical flow pretrained model (may be single stream OF model)")
    parser.add_argument('--two_stream_mode', type=TwoStreamMode, choices=list(TwoStreamMode),
                        help='spatial/ temporal/ spatial_temporal')
    parser.add_argument('--use_memcached', action='store_true', help='whether use memcached to boost speed of fetch crop&mask') #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--use_paper_num_label", action="store_true", help="only to use paper reported number of labels"
                                                                           " to train")
    parser.add_argument("--roi_align", action="store_true", help="whether to use roi align or roi pooling layer in CNN")
    parser.add_argument("--T", '-T', type=int, default=10, help="sequence length of one video clip")
    parser.add_argument("--out_channel",type=int, default=2048, help="length of extract ROI feature")
    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.pid, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    pid = str(os.getpid())
    pid_file_path = args.pid + os.sep + "{0}_{1}_fold_{2}.pid".format(args.database, args.fold, args.split_idx)
    with open(pid_file_path, "w") as file_obj:
        file_obj.write(pid)
        file_obj.flush()


    print('GPU: {}'.format(",".join(list(map(str, args.gpu)))))

    adaptive_AU_database(args.database)
    mc_manager = None
    if args.use_memcached:
        from collections_toolkit.memcached_manager import PyLibmcManager
        mc_manager = PyLibmcManager(args.memcached_host)
        if mc_manager is None:
            raise IOError("no memcached found listen in {}".format(args.memcached_host))

    paper_report_label, class_num = squeeze_label_num_report(args.database, args.use_paper_num_label)
    paper_report_label_idx = list(paper_report_label.keys())
    au_rcnn_train_chain_list = []
    if args.backbone == 'i3d':
        if args.two_stream_mode == TwoStreamMode.rgb:
            i3d_feature_backbone = I3DFeatureExtractor(modality='rgb')
            i3d_roi_head = I3DRoIHead(out_channel=args.out_channel, roi_size=7, spatial_scale=1/16., dropout_prob=0.)
            chainer.serializers.load_npz(args.pretrained_flow, i3d_feature_backbone)
            chainer.serializers.load_npz(args.pretrained_flow, i3d_roi_head)
            au_rcnn_train_chain_rgb = AU_RCNN_ROI_Extractor(i3d_feature_backbone, i3d_roi_head)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_rgb)
        elif args.two_stream_mode == TwoStreamMode.optical_flow:
            i3d_feature_backbone_flow = I3DFeatureExtractor(modality='flow')
            i3d_roi_head = I3DRoIHead(out_channel=args.out_channel, roi_size=7, spatial_scale=1 / 16., dropout_prob=0.)
            au_rcnn_train_chain_flow = AU_RCNN_ROI_Extractor(i3d_feature_backbone_flow, i3d_roi_head)
            chainer.serializers.load_npz(args.pretrained_flow, i3d_feature_backbone_flow)
            chainer.serializers.load_npz(args.pretrained_flow, i3d_roi_head)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_flow)
        elif args.two_stream_mode == TwoStreamMode.rgb_flow:
            i3d_feature_backbone = I3DFeatureExtractor(modality='rgb')
            i3d_roi_head_rgb = I3DRoIHead(out_channel=args.out_channel, roi_size=7, spatial_scale=1 / 16., dropout_prob=0.)
            chainer.serializers.load_npz(args.pretrained_rgb, i3d_feature_backbone)
            chainer.serializers.load_npz(args.pretrained_rgb, i3d_roi_head_rgb)
            au_rcnn_train_chain_rgb = AU_RCNN_ROI_Extractor(i3d_feature_backbone, i3d_roi_head_rgb)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_rgb)

            i3d_feature_backbone_flow = I3DFeatureExtractor(modality='flow')
            i3d_roi_head_flow = I3DRoIHead(out_channel=args.out_channel, roi_size=7, spatial_scale=1 / 16., dropout_prob=0.)
            au_rcnn_train_chain_flow = AU_RCNN_ROI_Extractor(i3d_feature_backbone_flow, i3d_roi_head_flow)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_flow)

            chainer.serializers.load_npz(args.pretrained_flow, i3d_feature_backbone_flow)
            chainer.serializers.load_npz(args.pretrained_flow, i3d_roi_head_flow)


    au_rcnn_train_loss = AU_RCNN_TrainChainLoss()
    loss_head_module = au_rcnn_train_loss
    model = Wrapper(au_rcnn_train_chain_list, loss_head_module, args.database, args.T, args.two_stream_mode, args.gpu)

    batch_size = args.batch_size
    img_dataset = AUDataset(database=args.database,
                           fold=args.fold, split_name='trainval',
                           split_index=args.split_idx, mc_manager=mc_manager,
                           train_all_data=False)

    train_video_data = AU_video_dataset(au_image_dataset=img_dataset,
                                        sample_frame=args.T, train_mode=True,
                                        paper_report_label_idx=paper_report_label_idx)

    Transform = Transform3D
    substract_mean = SubStractMean(args.mean)
    train_video_data = TransformDataset(train_video_data, Transform(substract_mean, mirror=False))

    if args.proc_num == 1:
        train_iter = SerialIterator(train_video_data, batch_size * args.sample_frame, repeat=True, shuffle=False)
    else:
        train_iter = MultiprocessIterator(train_video_data,  batch_size=batch_size * args.sample_frame,
                                          n_processes=args.proc_num,
                                      repeat=True, shuffle=False, n_prefetch=10, shared_mem=10000000)


    for gpu in args.gpu:
        chainer.cuda.get_device_from_id(gpu).use()

    optimizer = None
    if args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad(lr=args.lr)  # 原本为MomentumSGD(lr=args.lr, momentum=0.9) 由于loss变为nan问题，改为AdaGrad
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop(lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    elif args.optimizer == "AdaDelta":
        optimizer = chainer.optimizers.AdaDelta()


    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    optimizer_name = args.optimizer


    key_str = "{0}_fold_{1}".format(args.fold, args.split_idx)
    file_list = []
    file_list.extend(os.listdir(args.out))
    # BP4D_3_fold_1_resnet101@rnn@no_temporal@use_paper_num_label@roi_align@label_dep_layer@conv_lstm@sampleframe#13_model.npz
    use_paper_key_str = "use_paper_num_label" if args.use_paper_num_label else "all_{}_label".format(args.database)
    roi_align_key_str = "roi_align" if args.roi_align else "roi_pooling"

    single_model_file_name = args.out + os.sep + \
                             '{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@sampleframe#{7}_model.npz'.format(args.database,
                                                                                args.fold, args.split_idx,
                                                                                args.backbone, args.two_stream_mode,
                                                                                use_paper_key_str, roi_align_key_str,
                                                                                 args.T)
    print(single_model_file_name)
    pretrained_optimizer_file_name = args.out + os.sep +\
                             '{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@sampleframe#{7}_optimizer.npz'.format(args.database,
                                                                                args.fold, args.split_idx,
                                                                                args.backbone, args.two_stream_mode,
                                                                                use_paper_key_str, roi_align_key_str,
                                                                                 args.T)
    print(pretrained_optimizer_file_name)

    if os.path.exists(pretrained_optimizer_file_name):
        print("loading optimizer snatshot:{}".format(pretrained_optimizer_file_name))
        chainer.serializers.load_npz(pretrained_optimizer_file_name, optimizer)

    if os.path.exists(single_model_file_name):
        print("loading pretrained snapshot:{}".format(single_model_file_name))
        chainer.serializers.load_npz(single_model_file_name, model)


    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu[0],
                          converter=lambda batch, device: concat_examples(batch, device, padding=0))

    @training.make_extension(trigger=(1, "epoch"))
    def reset_order(trainer):
        print("reset dataset order after one epoch")
        trainer.updater._iterators["main"].dataset._dataset.reset_for_train_mode()

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(reset_order)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=os.path.basename(pretrained_optimizer_file_name)),
        trigger=(args.snapshot, 'iteration'))

    log_interval = 100, 'iteration'
    print_interval = 10, 'iteration'
    plot_interval = 10, 'iteration'
    if args.optimizer != "Adam" and args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1),
                       trigger=(10, 'epoch'))
    elif args.optimizer == "Adam":
        trainer.extend(chainer.training.extensions.ExponentialShift("alpha", 0.1, optimizer=optimizer), trigger=(10, 'epoch'))
    if args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_interval,
                                            log_name="log_{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@sampleframe#{7}.log".format(
                                                                                args.database,
                                                                                args.fold, args.split_idx,
                                                                                args.backbone, args.two_stream_mode,
                                                                                use_paper_key_str, roi_align_key_str,
                                                                                args.T)))
    # trainer.reporter.add_observer("main_par", model.loss_head_module)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss','main/accuracy',
         ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=100))

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/loss'],
                file_name='loss_{0}_fold_{1}_{2}@{3}@{4}@{5}.png'.format(args.fold, args.split_idx,
                                                                                         args.backbone, args.spatial_edge_mode,
                                                                                          args.temporal_edge_mode,args.conv_rnn_type), trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/accuracy'],
                file_name='accuracy_{0}_fold_{1}_{2}@{3}@{4}@{5}.png'.format(args.fold, args.split_idx,
                                                                                         args.backbone, args.spatial_edge_mode,
                                                                                          args.temporal_edge_mode,args.conv_rnn_type), trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.run()
    # cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()




if __name__ == '__main__':
    main()
