#!/usr/local/anaconda3/bin/python3
from __future__ import division

import cv2
from multiprocess.pool import Pool
import sys


sys.path.insert(0, '/home1/machen/face_expr')
from img_toolkit.face_mask_cropper import FaceMaskCropper

from space_time_AU_rcnn.datasets.AU_dataset import AUDataset
from space_time_AU_rcnn.datasets.parallel_tools import parallel_landmark_and_conn_component

try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

import argparse
import os
import sys
sys.path.insert(0, '/home1/machen/face_expr')

from space_time_AU_rcnn.constants.enum_type import SpatialEdgeMode, RecurrentType
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import config


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]

def main():
    parser = argparse.ArgumentParser(
        description='Space Time Action Unit R-CNN training example:')
    parser.add_argument('--pid', '-pp', default='/tmp/SpaceTime_AU_R_CNN/')
    parser.add_argument('--gpu', '-g', nargs='+', type=int, help='GPU ID, multiple GPU split by space')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--out', '-o', default='end_to_end_result',
                        help='Output directory')
    parser.add_argument('--trainval',  default='train',
                        help='train/test')
    parser.add_argument('--database',  default='BP4D',
                        help='Output directory: BP4D/DISFA/BP4D_DISFA')
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--snapshot', '-snap', type=int, default=1000)
    parser.add_argument('--need_validate', action='store_true', help='do or not validate during training')
    parser.add_argument('--mean', default=config.ROOT_PATH+"BP4D/idx/mean_no_enhance.npy", help='image mean .npy file')
    parser.add_argument('--backbone', default="mobilenet_v1", help="vgg/resnet101/mobilenet_v1 for train")
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: RMSprop/AdaGrad/Adam/SGD/AdaDelta')
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
    parser.add_argument("--previous_frame", type=int, default=50)
    parser.add_argument("--sample_frame", '-sample', type=int, default=25)
    parser.add_argument("--snap_individual", action="store_true", help="whether to snapshot each individual epoch/iteration")
    parser.add_argument("--proc_num", "-proc", type=int, default=1)
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


    train_data = AUDataset(database=args.database,
                           fold=args.fold, split_name=args.trainval,
                           split_index=args.split_idx, mc_manager=mc_manager, train_all_data=False,
                           )
    result_data = [img_path for img_path, AU_set, current_database_name in train_data.result_data
                   if args.database + "|" + img_path not in mc_manager]
    sub_list = split_list(result_data, len(result_data)//100)

    for img_path_lst in sub_list:
        with Pool(processes=50) as pool:
            input_list = [(img_path, None, None) for img_path in img_path_lst]
            result =\
                pool.starmap(parallel_landmark_and_conn_component, input_list)
            pool.close()
            pool.join()
            for img_path, AU_box_dict, landmark_dict, box_is_whole_image in result:
                key_prefix = args.database + "|"
                key = key_prefix + img_path
                orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                new_face, rect = FaceMaskCropper.dlib_face_crop(orig_img, landmark_dict)

                print("write {}".format(key))
                if mc_manager is not None and key not in mc_manager:
                    save_dict = {"landmark_dict": landmark_dict, "AU_box_dict": AU_box_dict, "crop_rect":rect}
                    mc_manager.set(key, save_dict)




if __name__ == '__main__':
    main()
