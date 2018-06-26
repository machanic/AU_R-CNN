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
from CNN.model.train_chain import TrainChain
from CNN.model.resnet101 import ResNet
from CNN import transforms

from CNN.datasets.image_dataset import ImageDataset
from chainer.dataset import concat_examples
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import config
from chainer.iterators import MultiprocessIterator, SerialIterator
from CNN.extensions.AU_evaluator import AUEvaluator
import json

class OccludeTransform(object):
    def __init__(self, occlude):
        self.occlude = occlude
    def __call__(self, in_data):
        img, label = in_data
        if self.occlude == "upper":
            img[:, img.shape[1] // 2:, :] = 0
        elif self.occlude == "lower":
            img[:, :img.shape[1] // 2, :] = 0
        elif self.occlude == "left":
            img[:, :, img.shape[1] // 2:] = 0
        elif self.occlude == "right":
            img[:, :, :img.shape[1] // 2] = 0
        return img, label


class Transform(object):

    def __init__(self, mean_rgb_path, mirror=True):
        self.mirror = mirror
        self.mean_rgb = np.load(mean_rgb_path)
        self.mean_rgb = self.mean_rgb.astype(np.float32)
        self.mean_rgb = np.resize(self.mean_rgb, (3, config.IMG_SIZE[0], config.IMG_SIZE[1]))
        # self.mean_flow = np.tile(np.expand_dims(np.load(mean_flow_path), axis=0), reps=(L, 1, 1, 1))[:, :2, :, :]
        # self.mean_flow = self.mean_flow.astype(np.float32)

    def __call__(self, in_data):
        img,  label = in_data
        _, H, W = img.shape
        img = img - self.mean_rgb
        # horizontally flip and random shift box
        if self.mirror:
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
        return img, label


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
    print("chainer cudnn enabled: {}".format(chainer.cuda.cudnn_enabled))
    parser = argparse.ArgumentParser(
        description='Action Unit R-CNN training example:')
    parser.add_argument('--pid', '-pp', default='/tmp/AU_R_CNN/')
    parser.add_argument('--gpu', '-g', default="0", help='GPU ID, multiple GPU split by comma, \ '
                                                         'Note that BPTT updater do not support multi-GPU')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--database',  default='BP4D',
                        help='Output directory: BP4D/DISFA/BP4D_DISFA')
    parser.add_argument('--readtype', default='rgb',
                        help='rgb/flow')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=20)
    parser.add_argument('--snapshot', '-snap', type=int, default=1000)
    parser.add_argument('--mean', default=config.ROOT_PATH+"BP4D/idx/mean_rgb.npy", help='image mean .npy file')
    parser.add_argument('--feature_model', default="resnet101", help="vgg or resnet101 for train")
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
    parser.add_argument('--pretrained_model', default='resnet101', help='imagenet/vggface/resnet101/*.npz')
    parser.add_argument('--use_memcached', action='store_true', help='whether use memcached to boost speed of fetch crop&mask') #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    parser.add_argument("--is_pretrained", action="store_true", help="whether is to pretrain BP4D later will for DISFA dataset or not")
    parser.add_argument("--pretrained_target", '-pt', default="", help="whether pretrain label set will use DISFA or not")
    parser.add_argument('--eval_mode', action='store_true', help='Use test datasets for evaluation metric')
    parser.add_argument('--test_model', default="", help='test model for evaluation')
    parser.add_argument('--occlude', default='',
                        help='whether to use occlude face of upper/left/right/lower/none to test')
    parser.add_argument("--img_resolution", type=int, default=512)
    args = parser.parse_args()
    config.IMG_SIZE = (args.img_resolution, args.img_resolution)
    if not os.path.exists(args.pid):
        os.makedirs(args.pid)
    pid = str(os.getpid())
    pid_file_path = args.pid + os.path.sep + "{0}_{1}_fold_{2}.pid".format(args.database, args.fold, args.split_idx)
    with open(pid_file_path, "w") as file_obj:
        file_obj.write(pid)
        file_obj.flush()

    print('GPU: {}'.format(args.gpu))
    if args.is_pretrained:
        adaptive_AU_database(args.pretrained_target)
    else:
        adaptive_AU_database(args.database)
    np.random.seed(args.seed)
    # 需要先构造一个list的txt文件:id_trainval_0.txt, 每一行是subject + "/" + emotion_seq + "/" frame
    mc_manager = None
    if args.use_memcached:
        from collections_toolkit.memcached_manager import PyLibmcManager
        mc_manager = PyLibmcManager(args.memcached_host)
        if mc_manager is None:
            raise IOError("no memcached found listen in {}".format(args.memcached_host))
    resnet101 = ResNet(len(config.AU_SQUEEZE), pretrained_model=args.pretrained_model)
    model = TrainChain(resnet101)

    if args.eval_mode:
        with chainer.no_backprop_mode(), chainer.using_config("train",False):
            if args.occlude:
                test_data = ImageDataset(database=args.database, fold=args.fold,
                                         split_name='test', split_index=args.split_idx, mc_manager=mc_manager,
                                         train_all_data=False, pretrained_target=args.pretrained_target,
                                         img_resolution=args.img_resolution)
                test_data = TransformDataset(test_data, Transform(mean_rgb_path=args.mean, mirror=False))
                assert args.occlude in ["upper","lower", "left", "right"]
                test_data = TransformDataset(test_data, OccludeTransform(args.occlude))

                if args.proc_num == 1:
                    test_iter = SerialIterator(test_data, 1, repeat=False, shuffle=True)
                else:
                    test_iter = MultiprocessIterator(test_data, batch_size=1, n_processes=args.proc_num,
                                                     repeat=False, shuffle=True,
                                                     n_prefetch=10, shared_mem=10000000)
                single_model_file_name = args.test_model
                chainer.serializers.load_npz(single_model_file_name, resnet101)
                gpu = int(args.gpu)
                chainer.cuda.get_device_from_id(gpu).use()
                resnet101.to_gpu(gpu)
                evaluator = AUEvaluator(test_iter, resnet101,
                                        lambda batch, device: concat_examples(batch, device, padding=0),
                                        args.database, "/home/machen/face_expr", device=gpu, npz_out_path=args.out
                                        + os.path.sep + "npz_occlude_{0}_split_{1}.npz".format(args.occlude, args.split_idx))
                observation = evaluator.evaluate()
                with open(
                        args.out + os.path.sep + "evaluation_occlude_{0}_fold_{1}_result_test_mode.json".format(args.occlude,
                                                                                                           args.split_idx),
                        "w") as file_obj:
                    file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
                    file_obj.flush()
            else:
                test_data = ImageDataset(database=args.database, fold=args.fold,
                                         split_name='test', split_index=args.split_idx, mc_manager=mc_manager,
                                         train_all_data=False, pretrained_target=args.pretrained_target, img_resolution=args.img_resolution)
                test_data = TransformDataset(test_data, Transform(mean_rgb_path=args.mean, mirror=False))
                if args.proc_num == 1:
                    test_iter = SerialIterator(test_data, 1, repeat=False, shuffle=False)
                else:
                    test_iter = MultiprocessIterator(test_data, batch_size=1, n_processes=args.proc_num,
                                                     repeat=False, shuffle=False,
                                                     n_prefetch=10, shared_mem=10000000)
                single_model_file_name = args.test_model
                chainer.serializers.load_npz(single_model_file_name, resnet101)

                gpu = int(args.gpu) if "," not in args.gpu else int(args.gpu[:args.gpu.index(",")])
                chainer.cuda.get_device_from_id(gpu).use()
                resnet101.to_gpu(gpu)
                evaluator = AUEvaluator(test_iter, resnet101,
                                        lambda batch, device: concat_examples(batch, device, padding=0),
                                        args.database, "/home/machen/face_expr", device=gpu, npz_out_path=args.out
                                                                + os.path.sep + "npz_split_{}.npz".format(args.split_idx))
                observation = evaluator.evaluate()
                with open(args.out + os.path.sep + "evaluation_split_{}_result_train_mode.json".format(args.split_idx),
                          "w") as file_obj:
                    file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
                    file_obj.flush()
        return


    train_data = ImageDataset(database=args.database,
                           fold=args.fold, split_name='trainval',
                           split_index=args.split_idx, mc_manager=mc_manager, train_all_data=args.is_pretrained,
                            read_type=args.readtype,
                           pretrained_target=args.pretrained_target, img_resolution=args.img_resolution
                           )
    train_data = TransformDataset(train_data, Transform(args.mean, mirror=True))
    if args.proc_num == 1:
        train_iter = SerialIterator(train_data, args.batch_size, True, True)
    else:
        train_iter = MultiprocessIterator(train_data,  batch_size=args.batch_size, n_processes=args.proc_num,
                                      repeat=True, shuffle=True, n_prefetch=10,shared_mem=31457280)

    if "," in args.gpu:
        for gpu in args.gpu.split(","):
            chainer.cuda.get_device_from_id(int(gpu)).use()
    else:
        chainer.cuda.get_device_from_id(int(args.gpu)).use()

    optimizer = None
    if args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad(lr=args.lr)  # 原本为MomentumSGD(lr=args.lr, momentum=0.9) 由于loss变为nan问题，改为AdaGrad
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop(lr=args.lr)
    elif args.optimizer == 'Adam':
        print("using Adam")
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    elif args.optimizer == "AdaDelta":
        print("using AdaDelta")
        optimizer = chainer.optimizers.AdaDelta()


    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    optimizer_name = args.optimizer

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    pretrained_optimizer_file_name = '{0}_{1}_fold_{2}_{3}_{4}_optimizer.npz'.format(args.database, args.fold, args.split_idx,
                                                                                     args.feature_model,
                                                                                     optimizer_name)
    pretrained_optimizer_file_name = args.out + os.path.sep + pretrained_optimizer_file_name

    single_model_file_name = args.out + os.path.sep + '{0}_{1}_fold_{2}_{3}_model.npz'.format(args.database,
                                                                                          args.fold, args.split_idx,
                                                                                         args.feature_model)

    if os.path.exists(pretrained_optimizer_file_name):
        print("loading optimizer snatshot:{}".format(pretrained_optimizer_file_name))
        chainer.serializers.load_npz(pretrained_optimizer_file_name, optimizer)


    if os.path.exists(single_model_file_name):
        print("loading pretrained snapshot:{}".format(single_model_file_name))
        chainer.serializers.load_npz(single_model_file_name, model.backbone)


    print(" GPU({0}) updater".format(args.gpu))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=int(args.gpu),
                          converter=lambda batch, device: concat_examples(batch, device, padding=0))

    trainer = training.Trainer(
            updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=os.path.basename(pretrained_optimizer_file_name)),
        trigger=(args.snapshot, 'iteration'))

    trainer.extend(
        chainer.training.extensions.snapshot_object(model.backbone,
                                                    filename=os.path.basename(single_model_file_name)),
        trigger=(args.snapshot, 'iteration'))

    log_interval = 100, 'iteration'
    print_interval = 100, 'iteration'
    plot_interval = 100, 'iteration'
    if args.optimizer != "Adam" and args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1),
                       trigger=(10, 'epoch'))
    elif args.optimizer == "Adam":
        # use Adam
        trainer.extend(chainer.training.extensions.ExponentialShift("alpha", 0.5, optimizer=optimizer), trigger=(10, 'epoch'))
    if args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_interval,log_name="{0}_fold_{1}.log".format(args.fold, args.split_idx)))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss','main/accuracy',
         ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=100))

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/loss',"validation/main/loss"],
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





if __name__ == '__main__':
    main()
