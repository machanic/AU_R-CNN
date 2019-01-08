#!/usr/local/anaconda3/bin/python3
from __future__ import division
import sys
sys.path.insert(0, '/home/machen/face_expr')

from AU_rcnn.extensions.speed_evaluator import SpeedEvaluator


from AU_rcnn.links.model.faster_rcnn.faster_rcnn_vgg19 import FasterRCNNVGG19
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_mobilenet_v1 import FasterRCNN_MobilenetV1

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
from AU_rcnn.links.model.faster_rcnn import FasterRCNNTrainChain, FasterRCNNVGG16, FasterRCNNResnet101
from AU_rcnn import transforms

from AU_rcnn.datasets.AU_dataset import AUDataset
from chainer.dataset import concat_examples
from AU_rcnn.extensions.special_converter import concat_examples_not_none
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import config
from chainer.training import extensions
from chainer.iterators import MultiprocessIterator, SerialIterator
from AU_rcnn.extensions.AU_evaluator import AUEvaluator
from AU_rcnn.links.model.faster_rcnn.feature_pyramid_network import FPN101
from AU_rcnn.links.model.faster_rcnn.feature_pyramid_train_chain import FPNTrainChain
import json
import os
# new feature support:
# 1. 支持resnet101/resnet50/VGG的模块切换;  2.支持LSTM/Linear的切换(LSTM用在score前的中间层); 3.支持多GPU切换
# 4. 支持指定最终用于提取的FC层的输出向量长度， 5.支持是否进行validate（每制定epoch的时候）
# 6. 支持读取pretrained model从vgg_face或者imagenet的weight 7. 支持优化算法的切换，比如AdaGrad或RMSprop
# 8. 使用memcached

class OccludeTransform(object):

    def __init__(self, occlude):
        self.occlude = occlude
    def __call__(self, in_data):
        img, bbox, label = in_data
        if self.occlude == "upper":
            img[:, img.shape[1] // 2:, :] = 0
        elif self.occlude == "lower":
            img[:, :img.shape[1] // 2, :] = 0
        elif self.occlude == "left":
            img[:, :, img.shape[1] // 2:] = 0
        elif self.occlude == "right":
            img[:, :, :img.shape[1] // 2] = 0
        return img, bbox, label


class FakeBoxTransform(object):
    def __init__(self, database):
        self.database = database

    def __call__(self, in_data):
        img, _, label = in_data  # bbox shape = (9,4)
        bbox = np.asarray(config.FAKE_BOX[self.database], dtype=np.float32)  # replace fake box, because it is only for prediction, the order doesn't matter
        return img, bbox, label


class Transform(object):

    def __init__(self, faster_rcnn, mirror=True):
        self.faster_rcnn = faster_rcnn
        self.mirror = mirror

    def __call__(self, in_data):

        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
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
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=20)
    parser.add_argument('--snapshot', '-snap', type=int, default=1000)
    parser.add_argument('--need_validate', action='store_true', help='do or not validate during training')
    parser.add_argument('--mean', default=config.ROOT_PATH+"BP4D/idx/mean_rgb.npy", help='image mean .npy file')
    parser.add_argument('--feature_model', default="resnet101", help="vgg16/vgg19/resnet101 for train")
    parser.add_argument('--extract_len', type=int, default=1000)
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
    parser.add_argument('--pretrained_model', default='resnet101', help='imagenet/vggface/resnet101/*.npz')
    parser.add_argument('--pretrained_model_args', nargs='+', type=float,
                        help='you can pass in "1.0 224" or "0.75 224"')
    parser.add_argument('--use_memcached', action='store_true', help='whether use memcached to boost speed of fetch crop&mask') #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--snap_individual", action="store_true", help="whether to snapshot each individual epoch/iteration")
    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    parser.add_argument("--use_sigmoid_cross_entropy", "-sigmoid", action="store_true",
                        help="whether to use sigmoid cross entropy or softmax cross entropy")
    parser.add_argument("--is_pretrained", action="store_true", help="whether is to pretrain BP4D later will for DISFA dataset or not")
    parser.add_argument("--pretrained_target", '-pt', default="", help="whether pretrain label set will use DISFA or not")
    parser.add_argument("--fix", '-fix', action="store_true", help="whether to fix first few conv layers or not")
    parser.add_argument('--occlude',  default='',
                        help='whether to use occlude face of upper/left/right/lower/none to test')
    parser.add_argument("--prefix", '-prefix', default="", help="_beta, for example 3_fold_beta")
    parser.add_argument('--eval_mode', action='store_true', help='Use test datasets for evaluation metric')
    parser.add_argument("--img_resolution", type=int, default=512)
    parser.add_argument("--FERA", action='store_true', help='whether to use FERA data split train and validate')
    parser.add_argument('--FPN', action="store_true", help="whether to use feature pyramid network for training and prediction")
    parser.add_argument('--fake_box', action="store_true", help="whether to use fake average box coordinate to predict")
    parser.add_argument('--roi_align', action="store_true",
                        help="whether to use roi_align or roi_pooling")
    parser.add_argument("--train_test", default="trainval", type=str)
    parser.add_argument("--trail_times", default=20, type=int)
    parser.add_argument("--each_trail_iteration", default=1000, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.pid):
        os.makedirs(args.pid)
    pid = str(os.getpid())
    pid_file_path = args.pid + os.sep + "{0}_{1}_fold_{2}.pid".format(args.database, args.fold, args.split_idx)
    # with open(pid_file_path, "w") as file_obj:
    #     file_obj.write(pid)
    #     file_obj.flush()

    config.IMG_SIZE = (args.img_resolution, args.img_resolution)

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

    if args.FPN:
        faster_rcnn = FPN101(len(config.AU_SQUEEZE), pretrained_resnet=args.pretrained_model, use_roialign=args.roi_align,
                             mean_path=args.mean,min_size=args.img_resolution,max_size=args.img_resolution)
    elif args.feature_model == 'vgg16':
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model=args.pretrained_model,
                                      mean_file=args.mean,
                                       min_size=args.img_resolution,max_size=args.img_resolution,
                                      extract_len=args.extract_len, fix=args.fix)  # 可改为/home/nco/face_expr/result/snapshot_model.npz
    elif args.feature_model == 'vgg19':
        faster_rcnn = FasterRCNNVGG19(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model=args.pretrained_model,
                                      mean_file=args.mean,
                                      min_size=args.img_resolution, max_size=args.img_resolution,
                                      extract_len=args.extract_len, dataset=args.database, fold=args.fold, split_idx=args.split_idx)
    elif args.feature_model == 'resnet101':
        faster_rcnn = FasterRCNNResnet101(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model=args.pretrained_model,
                                      mean_file=args.mean,  min_size=args.img_resolution,max_size=args.img_resolution,
                                      extract_len=args.extract_len)  # 可改为/home/nco/face_expr/result/snapshot_model.npz
    elif args.feature_model == "mobilenet_v1":
        faster_rcnn = FasterRCNN_MobilenetV1(pretrained_model_type=args.pretrained_model_args,
                                      min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                      mean_file=args.mean,  n_class=len(config.AU_SQUEEZE)
                                      )

    batch_size = args.batch_size

    with chainer.no_backprop_mode(), chainer.using_config("train",False):

        test_data = AUDataset(database=args.database, fold=args.fold, img_resolution=args.img_resolution,
                              split_name=args.train_test, split_index=args.split_idx, mc_manager=mc_manager,
                               train_all_data=False, prefix=args.prefix,
                              pretrained_target=args.pretrained_target, is_FERA=args.FERA)
        test_data = TransformDataset(test_data,
                                     Transform(faster_rcnn, mirror=False))
        if args.fake_box:
            test_data = TransformDataset(test_data, FakeBoxTransform(args.database))
        if args.proc_num == 1:
            test_iter = SerialIterator(test_data, args.batch_size, repeat=False, shuffle=True)
        else:
            test_iter = MultiprocessIterator(test_data, batch_size=args.batch_size, n_processes=args.proc_num,
                                             repeat=False, shuffle=True,
                                             n_prefetch=10, shared_mem=10000000)

        gpu = int(args.gpu) if "," not in args.gpu else int(args.gpu[:args.gpu.index(",")])
        chainer.cuda.get_device_from_id(gpu).use()
        faster_rcnn.to_gpu(gpu)
        evaluator = SpeedEvaluator(test_iter, faster_rcnn,
                                lambda batch, device: concat_examples_not_none(batch, device, padding=-99),
                                device=gpu, trail_times=args.trail_times,
                                   each_trail_iteration=args.each_trail_iteration, database=args.database)
        observation = evaluator.evaluate()
        with open(args.out + os.path.sep + "evaluation_speed_test.json", "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()




if __name__ == '__main__':
    main()
