#!/usr/local/anaconda3/bin/python3
from __future__ import division

import sys

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
from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_train_chain import AU_RCNN_ROI_Extractor, AU_RCNN_TrainChainLoss
from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_resnet101 import AU_RCNN_Resnet101
from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_vgg import AU_RCNN_VGG16
from two_stream_rgb_flow.model.wrap_model.wrapper import Wrapper
from two_stream_rgb_flow import transforms
from two_stream_rgb_flow.datasets.AU_video_dataset import AU_video_dataset
from two_stream_rgb_flow.datasets.AU_dataset import AUDataset
from two_stream_rgb_flow.constants.enum_type import TwoStreamMode
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

    def __init__(self, au_rcnn, mean_rgb_path, mean_flow_path, mirror=True):
        self.au_rcnn = au_rcnn
        self.mirror = mirror
        self.mean_rgb = np.load(mean_rgb_path)
        self.mean_flow = np.load(mean_flow_path)

    def __call__(self, in_data):
        rgb_img, flow_img, bbox, label = in_data
        rgb_img = self.au_rcnn.prepare(rgb_img, self.mean_rgb)
        flow_img = self.au_rcnn.prepare(flow_img, self.mean_flow)
        assert len(np.where(bbox < 0)[0]) == 0
        # horizontally flip and random shift box
        if self.mirror:
            rgb_img, params = transforms.random_flip(
                rgb_img, x_random=True, return_param=True)
            if params['x_flip']:
                flow_img = flow_img[:,:,::-1]
            bbox = transforms.flip_bbox(
                bbox, (rgb_img.shape[0], rgb_img.shape[1]), x_flip=params['x_flip'])

        return rgb_img, flow_img, bbox, label



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
        description='Space Time Action Unit R-CNN training example:')
    parser.add_argument('--pid', '-pp', default='/tmp/SpaceTime_AU_R_CNN/')
    parser.add_argument('--gpu', '-g', nargs='+', type=int, help='GPU ID, multiple GPU split by space')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--out', '-o', default='two_stream_out',
                        help='Output directory')
    parser.add_argument('--database',  default='BP4D',
                        help='Output directory: BP4D/DISFA/BP4D_DISFA')
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--snapshot', '-snap', type=int, default=1000)
    parser.add_argument('--mean_rgb', default=config.ROOT_PATH+"BP4D/idx/mean_rgb.npy", help='image mean .npy file')
    parser.add_argument('--mean_flow', default=config.ROOT_PATH+"BP4D/idx/mean_flow.npy", help='image mean .npy file')
    parser.add_argument('--backbone', default="mobilenet_v1", help="vgg/resnet101/mobilenet_v1 for train")
    parser.add_argument('--optimizer', default='SGD', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
    parser.add_argument('--pretrained_model_rgb', help='imagenet/mobilenet_v1/resnet101/*.npz')
    parser.add_argument('--pretrained_model_flow', help="path of optical flow pretrained model (may be single stream OF model)")
    parser.add_argument('--two_stream_mode', type=TwoStreamMode, choices=list(TwoStreamMode),
                        help='rgb_flow/ optical_flow/ rgb')
    parser.add_argument('--use_memcached', action='store_true', help='whether use memcached to boost speed of fetch crop&mask') #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument("--fix", action="store_true", help="fix parameter of conv2 update when finetune")
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--use_paper_num_label", action="store_true", help="only to use paper reported number of labels"
                                                                           " to train")
    parser.add_argument("--roi_align", action="store_true", help="whether to use roi align or roi pooling layer in CNN")
    parser.add_argument("--debug", action="store_true", help="debug mode for 1/50 dataset")
    parser.add_argument("--T", '-T', type=int, default=10)
    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    parser.add_argument("--fetch_mode", type=int, default=1)
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
    if args.backbone == 'vgg':
        au_rcnn = AU_RCNN_VGG16(pretrained_model=args.pretrained_model_rgb,
                                    min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                    use_roi_align=args.roi_align)
        au_rcnn_train_chain = AU_RCNN_ROI_Extractor(au_rcnn)
        au_rcnn_train_chain_list.append(au_rcnn_train_chain)
    elif args.backbone == 'resnet101':

        if args.two_stream_mode != TwoStreamMode.rgb_flow:
            assert (args.pretrained_model_rgb == "" and args.pretrained_model_flow != "") or\
                   (args.pretrained_model_rgb != "" and args.pretrained_model_flow == "")
            pretrained_model = args.pretrained_model_rgb if args.pretrained_model_rgb else args.pretrained_model_flow
            au_rcnn = AU_RCNN_Resnet101(pretrained_model=pretrained_model,
                                        min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                         classify_mode=True, n_class=class_num,
                                        use_roi_align=args.roi_align,
                                        use_optical_flow_input=(args.two_stream_mode == TwoStreamMode.optical_flow),
                                        temporal_length=args.T)
            au_rcnn_train_chain = AU_RCNN_ROI_Extractor(au_rcnn)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain)
        else: # rgb_flow mode
            au_rcnn_rgb = AU_RCNN_Resnet101(pretrained_model=args.pretrained_model_rgb,
                                            min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                             classify_mode=True, n_class=class_num,
                                            use_roi_align=args.roi_align,
                                            use_optical_flow_input=False, temporal_length=args.T)


            au_rcnn_optical_flow = AU_RCNN_Resnet101(pretrained_model=args.pretrained_model_flow,
                                                     min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                                      classify_mode=True,
                                                     n_class=class_num,
                                                     use_roi_align=args.roi_align,
                                                     use_optical_flow_input=True, temporal_length=args.T)
            au_rcnn_train_chain_rgb = AU_RCNN_ROI_Extractor(au_rcnn_rgb)
            au_rcnn_train_chain_optical_flow = AU_RCNN_ROI_Extractor(au_rcnn_optical_flow)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_rgb)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_optical_flow)
            au_rcnn = au_rcnn_rgb


    loss_head_module = AU_RCNN_TrainChainLoss()

    model = Wrapper(au_rcnn_train_chain_list, loss_head_module, args.database, args.T,
                    two_stream_mode=args.two_stream_mode, gpus=args.gpu)
    batch_size = args.batch_size
    img_dataset = AUDataset(database=args.database,
                           fold=args.fold, split_name='trainval',
                           split_index=args.split_idx, mc_manager=mc_manager,
                           train_all_data=False)

    train_video_data = AU_video_dataset(au_image_dataset=img_dataset,
                                        sample_frame=args.T, train_mode=(args.two_stream_mode != TwoStreamMode.optical_flow),
                                        paper_report_label_idx=paper_report_label_idx)

    Transform = Transform3D

    train_video_data = TransformDataset(train_video_data, Transform(au_rcnn, mirror=False,mean_rgb_path=args.mean_rgb,
                                                                    mean_flow_path=args.mean_flow))

    if args.proc_num == 1:
        train_iter = SerialIterator(train_video_data, batch_size * args.T, repeat=True, shuffle=False)
    else:
        train_iter = MultiprocessIterator(train_video_data,  batch_size=batch_size * args.T,
                                          n_processes=args.proc_num,
                                      repeat=True, shuffle=False, n_prefetch=10, shared_mem=10000000)

    if len(args.gpu) > 1:
        for gpu in args.gpu:
            chainer.cuda.get_device_from_id(gpu).use()
    else:
        chainer.cuda.get_device_from_id(args.gpu[0]).use()
        model.to_gpu(args.gpu[0])

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


    # BP4D_3_fold_1_resnet101@rnn@no_temporal@use_paper_num_label@roi_align@label_dep_layer@conv_lstm@sampleframe#13_model.npz
    use_paper_key_str = "use_paper_num_label" if args.use_paper_num_label else "all_avail_label"
    roi_align_key_str = "roi_align" if args.roi_align else "roi_pooling"

    single_model_file_name = args.out + os.sep + \
                             '{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@T#{7}_model.npz'.format(args.database,
                                                                                                 args.fold,
                                                                                                 args.split_idx,
                                                                                                 args.backbone,
                                                                                                 args.two_stream_mode,
                                                                                                 use_paper_key_str,
                                                                                                 roi_align_key_str,
                                                                                                 args.T)

    print(single_model_file_name)
    pretrained_optimizer_file_name = args.out + os.sep + \
                                     '{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@T#{7}_optimizer.npz'.format(
                                         args.database,
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

    if args.fix:
        au_rcnn = model.au_rcnn_train_chain.au_rcnn
        au_rcnn.extractor.conv1.W.update_rule.enabled = False
        au_rcnn.extractor.bn1.gamma.update_rule.enabled = False
        au_rcnn.extractor.bn1.beta.update_rule.enabled = False
        res2_names = ["a", "b1", "b2"]
        for res2_name in res2_names:
            if res2_name == "a":

                getattr(au_rcnn.extractor.res2, res2_name).conv1.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn1.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn1.beta.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).conv2.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).conv3.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).conv4.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn2.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn2.beta.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn3.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn3.beta.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn4.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn4.beta.update_rule.enabled = False
            elif res2_name.startswith("b"):
                getattr(au_rcnn.extractor.res2, res2_name).conv1.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn1.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn1.beta.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).conv2.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).conv3.W.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn2.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn2.beta.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn3.gamma.update_rule.enabled = False
                getattr(au_rcnn.extractor.res2, res2_name).bn3.beta.update_rule.enabled = False


    # if (args.spatial_edge_mode in [SpatialEdgeMode.ld_rnn, SpatialEdgeMode.bi_ld_rnn] or args.temporal_edge_mode in \
    #     [TemporalEdgeMode.ld_rnn, TemporalEdgeMode.bi_ld_rnn]) or (args.conv_rnn_type != ConvRNNType.conv_rcnn):
    #     updater = BPTTUpdater(train_iter, optimizer, converter=lambda batch, device: concat_examples(batch, device,
    #                           padding=0), device=args.gpu[0])

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

    trainer.extend(
        chainer.training.extensions.snapshot_object(model,
                                                    filename=os.path.basename(single_model_file_name)),
        trigger=(args.snapshot, 'iteration'))

    log_interval = 100, 'iteration'
    print_interval = 100, 'iteration'
    plot_interval = 10, 'iteration'
    if args.optimizer != "Adam" and args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1),
                       trigger=(10, 'epoch'))
    elif args.optimizer == "Adam":
        trainer.extend(chainer.training.extensions.ExponentialShift("alpha", 0.1, optimizer=optimizer), trigger=(10, 'epoch'))
    if args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_interval,log_name="log_{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@T#{7}.log".format(
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
                file_name="loss_{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@T#{7}.png".format(
                                                                                args.database,
                                                                                args.fold, args.split_idx,
                                                                                args.backbone, args.two_stream_mode,
                                                                                use_paper_key_str, roi_align_key_str,
                                                                                args.T), trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/accuracy'],
                file_name="accuracy_{0}_{1}_fold_{2}_{3}@{4}@{5}@{6}@T#{7}.png".format(
                                                                                args.database,
                                                                                args.fold, args.split_idx,
                                                                                args.backbone, args.two_stream_mode,
                                                                                use_paper_key_str, roi_align_key_str,
                                                                                args.T), trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.run()
    # cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()




if __name__ == '__main__':
    main()
