import sys
sys.path.insert(0, "/home/machen/face_expr")
import argparse
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_resnet101 import FasterRCNNResnet101
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_vgg import FasterRCNNVGG16
from simple_graph_learning.dataset.AU_extractor_dataset import AUExtractorDataset
from simple_graph_learning.dataset.jump_exist_file_dataset import JumpExistFileDataset
from simple_graph_learning.iterators.batch_keep_order_iterator import BatchKeepOrderIterator
import config
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

import re
import os
import chainer
import numpy as np

def extract_mode(model_file_name):
    pattern = re.compile('.*(\d+)_fold_(\d+)_(.*?)_.*?\.npz', re.DOTALL)
    matcher = pattern.match(model_file_name)
    return_dict = {}
    if matcher:
        fold = matcher.group(1)
        split_idx = matcher.group(2)
        backbone = matcher.group(3)
        return_dict["fold"] = int(fold)
        return_dict["split_idx"] = int(split_idx)
        return_dict["backbone"] = backbone
    return return_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=-1,
                        help='each batch size will be a new file')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='gpu that used to extract feature')

    parser.add_argument("--out_dir", '-o', default="/home/machen/dataset/new_graph/")
    parser.add_argument("--model",'-m', help="the AU R-CNN pretrained model file to load to extract feature")
    parser.add_argument("--trainval_test", '-tt', help="train or test")
    parser.add_argument("--database", default="BP4D")
    parser.add_argument('--use_memcached', action='store_true',
                        help='whether use memcached to boost speed of fetch crop&mask')  #
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument('--force_write', action='store_true')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_no_enhance.npy",
                        help='image mean .npy file')
    parser.add_argument('--jump_exist_file', action='store_true',
                        help='image mean .npy file')
    args = parser.parse_args()

    adaptive_AU_database(args.database)
    mc_manager = None
    if args.use_memcached:
        from collections_toolkit.memcached_manager import PyLibmcManager
        mc_manager = PyLibmcManager(args.memcached_host)
        if mc_manager is None:
            raise IOError("no memcached found listen in {}".format(args.memcached_host))

    result_dict = extract_mode(args.model)
    fold = result_dict["fold"]
    backbone = result_dict["backbone"]
    split_idx = result_dict["split_idx"]
    if backbone == 'vgg':
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model="imagenet",
                                      mean_file=args.mean,
                                      use_lstm=False,
                                      extract_len=1000,
                                      fix=False)  # 可改为/home/nco/face_expr/result/snapshot_model.npz
    elif backbone == 'resnet101':
        faster_rcnn = FasterRCNNResnet101(n_fg_class=len(config.AU_SQUEEZE),
                                          pretrained_model=backbone,
                                          mean_file=args.mean,
                                          use_lstm=False,
                                          extract_len=1000, fix=False)
    assert os.path.exists(args.model)
    print("loading model file : {}".format(args.model))
    chainer.serializers.load_npz(args.model, faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        faster_rcnn.to_gpu(args.gpu)

    dataset = AUExtractorDataset(database=args.database,
                           fold=fold, split_name=args.trainval_test,
                           split_index=split_idx, mc_manager=mc_manager, use_lstm=False,
                           train_all_data=False,
                           prefix="", pretrained_target="", pretrained_model=faster_rcnn, extract_key="avg_pool",
                           device=-1, batch_size=args.batch_size
                           )
    train_test = "train" if args.trainval_test == "trainval" else "test"
    jump_dataset = JumpExistFileDataset(dataset, args.out_dir, fold, args.database, split_idx,
                                        args.batch_size, train_test, args.jump_exist_file)
    dataset_iter = BatchKeepOrderIterator(jump_dataset, batch_size=args.batch_size, repeat=False, shuffle=False)

    file_key_counter = 0
    last_sequence_key = None
    for batch in dataset_iter:
        features = []
        bboxes = []
        labels = []
        file_key_counter += 1
        for idx, (feature, bbox, label, img_path, _file_key_counter) in enumerate(batch):

            sequence_key = "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))
            if last_sequence_key is None:
                last_sequence_key = sequence_key
            if sequence_key!=last_sequence_key:
                file_key_counter = 1
                last_sequence_key = sequence_key
            assert file_key_counter == _file_key_counter, (file_key_counter, _file_key_counter, img_path)
            if feature is None:
                print("jump img_path : {}".format(img_path))
                continue

            features.extend(feature)
            bboxes.extend(bbox)
            labels.extend(label)
        if features:
            if args.trainval_test == "trainval":
                file_name = args.out_dir + os.sep + "{0}_{1}_fold_{2}".format(args.database,fold, split_idx) + "/train" +os.sep +sequence_key + "@" + str(file_key_counter) + ".npz"
            else:
                file_name = args.out_dir + os.sep + "{0}_{1}_fold_{2}".format(args.database,fold, split_idx) + "/test" + os.sep +sequence_key + "@" + str(file_key_counter) + ".npz"

            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            features = np.stack(features)
            bboxes = np.stack(bboxes)
            labels = np.stack(labels)
            print("write : {}".format(file_name))
            assert not os.path.exists(file_name), file_name
            np.savez(file_name, feature=features, bbox=bboxes, label=labels)



if __name__ == "__main__":
    main()