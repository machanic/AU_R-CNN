#!/usr/local/anaconda3/bin/python3
from __future__ import division
try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

import argparse
import random
import numpy as np
import os
import sys
sys.path.insert(0, '/home/machen/face_expr')
import chainer
from chainer import training

from chainer.datasets import TransformDataset
from AU_rcnn.links.model.faster_rcnn import FasterRCNNTrainChain, FasterRCNNVGG16, FasterRCNNResnet101
from AU_rcnn import transforms

from AU_rcnn.datasets.bp4d.AU_dataset_speed_optimized import AUDataset
from chainer.dataset import concat_examples
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from AU_rcnn.updater.update_bptt import BPTTUpdater
import config
from chainer.training import extensions
from AU_rcnn.iterator.remove_non_frame_iterator import RemoveNonLabelMultiprocessIterator
from chainer.iterators import MultiprocessIterator, SerialIterator
from AU_rcnn.extensions.AU_evaluator import AUEvaluator
import json
# new feature support:
# 1. 支持resnet101/resnet50/VGG的模块切换;  2.支持LSTM/Linear的切换(LSTM用在score前的中间层); 3.支持多GPU切换
# 4. 支持指定最终用于提取的FC层的输出向量长度， 5.支持是否进行validate（每制定epoch的时候）
# 6. 支持读取pretrained model从vgg_face或者imagenet的weight 7. 支持优化算法的切换，比如AdaGrad或RMSprop
# 8. 使用memcached
# *5.支持微信接口操作训练（回掉函数）用itchat

class Transform(object):

    def __init__(self, faster_rcnn, mirror=True, shift=False, use_lstm=False):
        self.faster_rcnn = faster_rcnn
        self.mirror = mirror
        self.use_lstm = use_lstm
        self.shift = shift

    def reset_state(self):
        self.faster_rcnn.head.reset_state()

    def __call__(self, in_data):
        if self.use_lstm:
            img, bbox, label, img_id = in_data
        else:
            img, bbox, label, AU_couple_lst = in_data
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
            if self.shift and not self.use_lstm:
                nonzero_label_idx_arr = np.unique(np.nonzero(label)[0])
                if len(nonzero_label_idx_arr) > 0:
                    random_shift_box = []
                    random_shift_label = []
                    # choice_idx_arr = np.random.choice(nonzero_label_idx_arr, size=len(nonzero_label_idx_arr),replace=False)
                    for idx, box in enumerate(bbox[nonzero_label_idx_arr]):
                        box_idx = nonzero_label_idx_arr[idx]
                        AU_couple = AU_couple_lst[box_idx]
                        current_shift = config.BOX_SHIFT[AU_couple]
                        y_min,x_min,y_max,x_max = box
                        y_min += random.randint(current_shift[0][0], current_shift[0][1])
                        x_min += random.randint(current_shift[1][0], current_shift[1][1])
                        y_max += random.randint(current_shift[2][0], current_shift[2][1])
                        x_max += random.randint(current_shift[3][0], current_shift[3][1])
                        random_box = np.asarray([y_min,x_min,y_max,x_max], dtype=np.float32)
                        random_box[random_box<0] = 0
                        random_box[random_box>img.shape[1]] = img.shape[1]
                        current_label = label[box_idx]
                        random_shift_box.append(random_box)
                        random_shift_label.append(current_label)
                    bbox = np.concatenate((bbox, np.asarray(random_shift_box)))
                    label = np.concatenate((label, np.asarray(random_shift_label)))
        if self.use_lstm:
            return img, bbox, label, img_id
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
    parser.add_argument('--mean', default=config.ROOT_PATH+"BP4D/idx/mean_no_enhance.npy", help='image mean .npy file')
    parser.add_argument('--feature_model', default="resnet101", help="vgg or resnet101 for train")
    parser.add_argument('--use_lstm', action='store_true', help='use LSTM or Linear in head module')  #LSTM 模式没办法用balance方法增加少类的box
    parser.add_argument('--extract_len', type=int, default=1000)
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
    parser.add_argument('--pretrained_model', default='resnet101', help='imagenet/vggface/resnet101/*.npz')
    parser.add_argument('--use_wechat', action='store_true', help='whether use wechat to control or not')
    parser.add_argument('--use_memcached', action='store_true', help='whether use memcached to boost speed of fetch crop&mask') #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument('--AU_count', default='AU_occr_count.dict', help="label balance dict file path for replicate for label balance")
    parser.add_argument('--random_shift','-shift', action='store_true', help='whether to random shift the bounding box or not')
    parser.add_argument("--fold", '-fd', type=int, default=3)
    parser.add_argument("--split_idx",'-sp', type=int, default=1)
    parser.add_argument("--snap_individual", action="store_true", help="whether to snapshot each individual epoch/iteration")
    parser.add_argument("--bptt_steps", '-bptt', type=int, default=20)
    parser.add_argument("--proc_num", "-proc", type=int, default=1)
    parser.add_argument("--use_sigmoid_cross_entropy", "-sigmoid", action="store_true",
                        help="whether to use sigmoid cross entropy or softmax cross entropy")
    parser.add_argument("--is_pretrained", action="store_true", help="whether is to pretrain BP4D later will for DISFA dataset or not")
    parser.add_argument("--pretrained_target", '-pt', default="", help="whether pretrain label set will use DISFA or not")
    parser.add_argument("--fix", '-fix', action="store_true", help="whether to fix first few conv layers or not")
    parser.add_argument("--prefix", '-prefix', default="", help="_beta, for example 3_fold_beta")
    parser.add_argument('--eval_mode', action='store_true', help='Use test datasets for evaluation metric')
    args = parser.parse_args()
    if not os.path.exists(args.pid):
        os.makedirs(args.pid)
    pid = str(os.getpid())
    pid_file_path = args.pid + os.sep + "{0}_{1}_fold_{2}.pid".format(args.database, args.fold, args.split_idx)
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

    if args.feature_model == 'vgg':
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model=args.pretrained_model,
                                      mean_file=args.mean,
                                      use_lstm=args.use_lstm,
                                      extract_len=args.extract_len, fix=args.fix)  # 可改为/home/nco/face_expr/result/snapshot_model.npz
    elif args.feature_model == 'resnet101':
        faster_rcnn = FasterRCNNResnet101(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model=args.pretrained_model,
                                      mean_file=args.mean,
                                      use_lstm=args.use_lstm,
                                      extract_len=args.extract_len, fix=args.fix)  # 可改为/home/nco/face_expr/result/snapshot_model.npz
    if args.use_lstm:
        faster_rcnn.reset_state()
    batch_size = args.batch_size if not args.use_lstm else 1

    if args.eval_mode:
        with chainer.no_backprop_mode():
            test_data = AUDataset(database=args.database, fold=args.fold,
                                          split_name='test', split_index=args.split_idx, mc_manager=mc_manager,
                                          use_lstm=args.use_lstm, train_all_data=False, prefix=args.prefix, pretrained_target=args.pretrained_target)
            test_data = TransformDataset(test_data, Transform(faster_rcnn, mirror=False, shift=False,use_lstm=args.use_lstm))
            if args.proc_num == 1:
                test_iter = SerialIterator(test_data, 1, repeat=False, shuffle=True)
            else:
                test_iter = MultiprocessIterator(test_data, batch_size=1, n_processes=args.proc_num,
                                                                   repeat=False, shuffle=True,
                                                                   n_prefetch=10, shared_mem=10000000)


            gpu = int(args.gpu) if "," not in args.gpu else int(args.gpu[:args.gpu.index(",")])
            chainer.cuda.get_device_from_id(gpu).use()
            faster_rcnn.to_gpu(gpu)
            evaluator = AUEvaluator(test_iter, faster_rcnn, lambda batch, device: concat_examples(batch, device, padding=-99), args.database,device=gpu)
            observation = evaluator.evaluate()
            with open(args.out + os.sep + "evaluation_result.json", "w") as file_obj:
                file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
                file_obj.flush()
        return




    train_data = AUDataset(database=args.database,
                           fold=args.fold, split_name='trainval',
                           split_index=args.split_idx, mc_manager=mc_manager, use_lstm=args.use_lstm, train_all_data=args.is_pretrained,
                           prefix=args.prefix, pretrained_target=args.pretrained_target
                           )


    train_data = TransformDataset(train_data, Transform(faster_rcnn,mirror=True,shift=args.random_shift,
                                                        use_lstm=args.use_lstm))

    # train_iter = chainer.iterators.SerialIterator(train_data, batch_size, repeat=True, shuffle=False)

    shuffle = True if not args.use_lstm else False
    if args.proc_num == 1:
        train_iter = SerialIterator(train_data, batch_size, True, shuffle)
    else:
        train_iter = MultiprocessIterator(train_data,  batch_size=batch_size, n_processes=args.proc_num,
                                      repeat=True, shuffle=shuffle, n_prefetch=10,shared_mem=31457280)

    model = FasterRCNNTrainChain(faster_rcnn, use_sigmoid_cross_entropy=args.use_sigmoid_cross_entropy,
                                 database=args.database, AU_count_name=args.AU_count)


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
    lstm_str = "lstm" if args.use_lstm else "linear"

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    pretrained_optimizer_file_name = '{0}_fold_{1}_{2}_{3}_{4}_optimizer.npz'.format(args.fold, args.split_idx,
                                                                                     args.feature_model,
                                                                                     lstm_str, optimizer_name)
    pretrained_optimizer_file_name = args.out + os.sep + pretrained_optimizer_file_name
    key_str = "{0}_fold_{1}".format(args.fold, args.split_idx)
    file_list = []
    if os.path.exists(args.out):
        file_list.extend(os.listdir(args.out))
    snapshot_model_file_name = args.out + os.sep + filter_last_checkpoint_filename(file_list, "model", key_str)
    single_model_file_name = args.out + os.sep + '{0}_fold_{1}_{2}_{3}_model.npz'.format(args.fold, args.split_idx,
                                                                                         args.feature_model, lstm_str)
    if os.path.exists(pretrained_optimizer_file_name):
        print("loading optimizer snatshot:{}".format(pretrained_optimizer_file_name))
        chainer.serializers.load_npz(pretrained_optimizer_file_name, optimizer)

    if args.snap_individual:
        if os.path.exists(snapshot_model_file_name) and os.path.isfile(snapshot_model_file_name):
            print("loading pretrained snapshot:{}".format(snapshot_model_file_name))
            chainer.serializers.load_npz(snapshot_model_file_name, model.faster_rcnn)
    else:
        if os.path.exists(single_model_file_name):
            print("loading pretrained snapshot:{}".format(single_model_file_name))
            chainer.serializers.load_npz(single_model_file_name, model.faster_rcnn)

    if args.use_lstm:
        updater = BPTTUpdater(train_iter, optimizer, args.bptt_steps, device=int(args.gpu),
                              converter=lambda batch, device: concat_examples(batch, device, padding=-99))
    elif "," in args.gpu:
        gpu_dict = {"main": int(args.gpu.split(",")[0])} # many gpu will use
        for slave_gpu in args.gpu.split(",")[1:]:
            gpu_dict[slave_gpu] = int(slave_gpu)
        updater = chainer.training.ParallelUpdater(train_iter, optimizer,
                                                   devices=gpu_dict,
                                                   converter=lambda batch, device: concat_examples(batch, device,
                                                                                                   padding=-99))
    else:
        print("only one GPU({0}) updater".format(args.gpu))
        updater = chainer.training.StandardUpdater(train_iter, optimizer, device=int(args.gpu),
                              converter=lambda batch, device: concat_examples(batch, device, padding=-99))

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(
        chainer.training.extensions.snapshot_object(optimizer,
                                                    filename=os.path.basename(pretrained_optimizer_file_name)),
        trigger=(args.snapshot, 'iteration'))

    if not args.snap_individual:
        trainer.extend(
            chainer.training.extensions.snapshot_object(model.faster_rcnn,
                                                        filename=single_model_file_name),
            trigger=(args.snapshot, 'iteration'))

    else:
        snap_model_file_name = '{0}_fold_{1}_{2}_{3}_model_snapshot_'.format(args.fold, args.split_idx,
                                                                                      args.feature_model, lstm_str)
        snap_model_file_name = snap_model_file_name+"{.updater.iteration}.npz"

        trainer.extend(
            chainer.training.extensions.snapshot_object(model.faster_rcnn,
                                                        filename=snap_model_file_name),
            trigger=(args.snapshot, 'iteration'))



    log_interval = 100, 'iteration'
    print_interval = 100, 'iteration'
    val_interval = 10000, "iteration"
    plot_interval = 100, 'iteration'
    if args.optimizer != "Adam" and args.optimizer != "AdaDelta":
        trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.5),
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
         'validation/main/loss','validation/main/accuracy'
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
                ['main/accuracy',"validation/main/accuracy"],
                file_name='accuracy_{0}_fold_{1}.png'.format(args.fold, args.split_idx), trigger=plot_interval
            ),
            trigger=plot_interval
        )

    if args.need_validate:
        print("need validate")

        validate_data = AUDataset(database=args.database, fold=args.fold,
                                  split_name='valid', split_index=args.split_idx, mc_manager=mc_manager,
                                  use_lstm=args.use_lstm, train_all_data=False, pretrained_target=args.pretrained_target)

        validate_data = TransformDataset(validate_data, Transform(faster_rcnn, mirror=False,shift=args.random_shift,
                                                                  use_lstm=args.use_lstm))

        if args.proc_num == 1:
            validate_iter = SerialIterator(validate_data, batch_size, repeat=False, shuffle=False)
        else:
            validate_iter = MultiprocessIterator(validate_data, batch_size=batch_size, n_processes=args.proc_num,
                                                           repeat=False, shuffle=False,
                                                           n_prefetch=10, shared_mem=31457280)

        gpu = int(args.gpu) if "," not in args.gpu else int(args.gpu[:args.gpu.index(",")])
        trainer.extend(extensions.Evaluator(iterator=validate_iter, target=model,
                                            converter=lambda batch, device: concat_examples(batch, device, padding=-99),
                                            device=gpu), trigger=val_interval)

    trainer.run()
    # cProfile.runctx("trainer.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()




if __name__ == '__main__':
    main()
