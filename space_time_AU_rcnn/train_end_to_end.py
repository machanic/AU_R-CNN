#!/usr/local/anaconda3/bin/python3
from __future__ import division

import cProfile
import pstats

try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

import argparse
import numpy as np
import os
import sys
sys.path.insert(0, '/home/machen/face_expr')
import chainer
from chainer import training

from chainer.datasets import TransformDataset
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_train_chain import AU_RCNN_ROI_Extractor, AU_RCNN_TrainChainLoss
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_resnet101 import AU_RCNN_Resnet101
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_vgg import AU_RCNN_VGG16
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_mobilenet_v1 import AU_RCNN_MobilenetV1
from space_time_AU_rcnn.model.roi_space_time_net.space_time_rnn import SpaceTimeRNN
from space_time_AU_rcnn.model.wrap_model.wrapper import Wrapper
from space_time_AU_rcnn import transforms
from space_time_AU_rcnn.datasets.AU_video_dataset import AU_video_dataset
from space_time_AU_rcnn.datasets.AU_dataset import AUDataset
from space_time_AU_rcnn.constants.enum_type import SpatialEdgeMode, RecurrentType
from chainer.dataset import concat_examples
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import config
from chainer.iterators import MultiprocessIterator, SerialIterator
from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report
from space_time_AU_rcnn.updater.partial_parallel_updater import PartialParallelUpdater

# new feature support:
# 1. 支持resnet101/resnet50/VGG的模块切换; 3.支持多GPU切换
# 5.支持是否进行validate（每制定epoch的时候）
# 6. 支持读取pretrained model从vgg_face或者imagenet的weight 7. 支持优化算法的切换，比如AdaGrad或RMSprop
# 8. 使用memcached

class Transform4D(object):

    def __init__(self, au_rcnn, mirror=True):
        self.au_rcnn = au_rcnn
        self.mirror = mirror

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, _, H, W = img.shape
        img = self.au_rcnn.prepare(img)
        _, _, o_H, o_W = img.shape
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        assert len(np.where(bbox < 0)[0]) == 0
        # horizontally flip and random shift box
        if self.mirror:
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label

class Transform3D(object):

    def __init__(self, au_rcnn, mirror=True):
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
    parser.add_argument('--out', '-o', default='end_to_end_result',
                        help='Output directory')
    parser.add_argument('--database',  default='BP4D',
                        help='Output directory: BP4D/DISFA/BP4D_DISFA')
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--snapshot', '-snap', type=int, default=1000)
    parser.add_argument('--need_validate', action='store_true', help='do or not validate during training')
    parser.add_argument('--mean', default=config.ROOT_PATH+"BP4D/idx/mean_no_enhance.npy", help='image mean .npy file')
    parser.add_argument('--backbone', default="mobilenet_v1", help="vgg/resnet101/mobilenet_v1 for train")
    parser.add_argument('--optimizer', default='SGD', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
    parser.add_argument('--pretrained_model', default='mobilenet_v1', help='imagenet/mobilenet_v1/resnet101/*.npz')
    parser.add_argument('--pretrained_model_args', nargs='+', type=float, help='you can pass in "1.0 224" or "0.75 224"')
    parser.add_argument('--spatial_edge_mode', type=SpatialEdgeMode, choices=list(SpatialEdgeMode),
                        help='1:all_edge, 2:configure_edge, 3:no_edge')
    parser.add_argument('--temporal_edge_mode', type=RecurrentType, choices=list(RecurrentType),
                        help='1:rnn, 2:attention_block, 3.point-wise feed forward(no temporal)')
    parser.add_argument("--bi_lstm", action="store_true", help="whether to use bi-lstm as Edge/Node RNN")
    parser.add_argument('--use_memcached', action='store_true', help='whether use memcached to boost speed of fetch crop&mask') #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--use_paper_num_label", action="store_true", help="only to use paper reported number of labels"
                                                                           " to train")
    parser.add_argument("--sample_frame", '-sample', type=int, default=10)
    parser.add_argument("--snap_individual", action="store_true", help="whether to snapshot each individual epoch/iteration")

    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    parser.add_argument("--fetch_mode", type=int, default=1)
    parser.add_argument("--au_rcnn_loss", action="store_true", help="whether to train AU R-CNN or not")
    parser.add_argument('--eval_mode', action='store_true', help='Use test datasets for evaluation metric')
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

    if args.backbone == 'vgg':
        au_rcnn = AU_RCNN_VGG16(pretrained_model=args.pretrained_model,
                                    min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                    mean_file=args.mean)
    elif args.backbone == 'resnet101':
        au_rcnn = AU_RCNN_Resnet101(pretrained_model=args.pretrained_model,
                                        min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                        mean_file=args.mean, classify_mode=args.au_rcnn_loss, n_class=class_num)
    elif args.backbone == "mobilenet_v1":
        au_rcnn = AU_RCNN_MobilenetV1(pretrained_model_type=args.pretrained_model_args,
                                      min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                      mean_file=args.mean, classify_mode=args.au_rcnn_loss, n_class=class_num
                                      )

    au_rcnn_train_chain = AU_RCNN_ROI_Extractor(au_rcnn)

    if args.au_rcnn_loss:
        au_rcnn_train_loss = AU_RCNN_TrainChainLoss()
        loss_head_module = au_rcnn_train_loss
    else:
        space_time_rnn = SpaceTimeRNN(args.database, args.layers, in_size=2048, out_size=class_num,
                         spatial_edge_model=args.spatial_edge_mode, recurrent_block_type=args.temporal_edge_mode,
                        bi_lstm=args.bi_lstm)
        loss_head_module = space_time_rnn
    model = Wrapper(au_rcnn_train_chain, loss_head_module, args.database, args.sample_frame)
    batch_size = args.batch_size
    img_dataset = AUDataset(database=args.database,
                           fold=args.fold, split_name='trainval',
                           split_index=args.split_idx, mc_manager=mc_manager,
                           train_all_data=False)

    train_video_data = AU_video_dataset(au_image_dataset=img_dataset,
                            sample_frame=args.sample_frame, train_mode=True,
                           paper_report_label_idx=paper_report_label_idx, fetch_use_parrallel_iterator=True)

    Transform = Transform3D  # FIXME

    train_video_data = TransformDataset(train_video_data, Transform(au_rcnn, mirror=True))

    if args.proc_num == 1:
        train_iter = SerialIterator(train_video_data, batch_size * args.sample_frame, repeat=True, shuffle=False)
    else:
        train_iter = MultiprocessIterator(train_video_data,  batch_size=batch_size * args.sample_frame, n_processes=args.proc_num,
                                      repeat=True, shuffle=False, n_prefetch=10,shared_mem=314572800)

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


    pretrained_optimizer_file_name = '{0}_fold_{1}_{2}_{3}_optimizer.npz'.format(args.fold, args.split_idx,
                                                                                     args.backbone, optimizer_name)
    pretrained_optimizer_file_name = args.out + os.sep + pretrained_optimizer_file_name
    key_str = "{0}_fold_{1}".format(args.fold, args.split_idx)
    file_list = []
    file_list.extend(os.listdir(args.out))
    snapshot_model_file_name = args.out + os.sep + filter_last_checkpoint_filename(file_list, "model", key_str)
    single_model_file_name = args.out + os.sep + '{0}_fold_{1}_{2}_model.npz'.format(args.fold, args.split_idx,
                                                                                         args.backbone)

    if os.path.exists(pretrained_optimizer_file_name):
        print("loading optimizer snatshot:{}".format(pretrained_optimizer_file_name))
        chainer.serializers.load_npz(pretrained_optimizer_file_name, optimizer)

    if args.snap_individual:
        if os.path.exists(snapshot_model_file_name) and os.path.isfile(snapshot_model_file_name):
            print("loading pretrained snapshot:{}".format(snapshot_model_file_name))
            chainer.serializers.load_npz(snapshot_model_file_name, model)
    else:
        if os.path.exists(single_model_file_name):
            print("loading pretrained snapshot:{}".format(single_model_file_name))
            chainer.serializers.load_npz(single_model_file_name, model)


    if len(args.gpu) > 1:
        gpu_dict = {"main": args.gpu[0]} # many gpu will use
        parallel_models = {"parallel": model.au_rcnn_train_chain}
        for slave_gpu in args.gpu[1:]:
            gpu_dict[slave_gpu] = int(slave_gpu)

        updater = PartialParallelUpdater(train_iter, optimizer, args.database, models=parallel_models,
                                                   devices=gpu_dict,
                                                   converter=lambda batch, device: concat_examples(batch, device,
                                                                                                   padding=0))
    else:
        print("only one GPU({0}) updater".format(args.gpu[0]))
        updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu[0],
                              converter=lambda batch, device: concat_examples(batch, device, padding=0))


    @training.make_extension(trigger=(1, "epoch"))
    def reset_order(trainer):
        print("reset dataset order after one epoch")
        trainer.updater._iterators["main"].dataset._dataset.reset()

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(reset_order)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=os.path.basename(pretrained_optimizer_file_name)),
        trigger=(args.snapshot, 'iteration'))

    if not args.snap_individual:
        snap_model_file_name = '{0}_fold_{1}_{2}_model.npz'.format(args.fold, args.split_idx,
                                                                             args.backbone)
        trainer.extend(
            chainer.training.extensions.snapshot_object(model,
                                                        filename=snap_model_file_name),
            trigger=(args.snapshot, 'iteration'))

    else:
        snap_model_file_name = '{0}_fold_{1}_{2}_model_snapshot_'.format(args.fold, args.split_idx,
                                                                                      args.backbone)
        snap_model_file_name = snap_model_file_name+"{.updater.iteration}.npz"

        trainer.extend(
            chainer.training.extensions.snapshot_object(model,
                                                        filename=snap_model_file_name),
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
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_interval,log_name="{0}_fold_{1}.log".format(args.fold,
                                                                                                                 args.split_idx)))
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
                file_name='loss_{0}_fold_{1}.png'.format(args.fold, args.split_idx), trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/accuracy'],
                file_name='accuracy_{0}_fold_{1}.png'.format(args.fold, args.split_idx), trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.run()
    # cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()




if __name__ == '__main__':
    main()
