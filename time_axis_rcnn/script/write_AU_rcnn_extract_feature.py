import sys
sys.path.insert(0, '/home/machen/face_expr')
from chainer.iterators import SerialIterator

from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report

import argparse
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_resnet101 import FasterRCNNResnet101
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_vgg import FasterRCNNVGG16

from lstm_end_to_end.model.AU_rcnn.au_rcnn_resnet101 import AU_RCNN_Resnet101
from lstm_end_to_end.model.wrap_model.wrapper import Wrapper
from lstm_end_to_end.model.AU_rcnn.au_rcnn_train_chain import AU_RCNN_ROI_Extractor
from lstm_end_to_end.constants.enum_type import TwoStreamMode

from time_axis_rcnn.datasets.extract_feature_dataset import FeatureExtractorDataset
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

def get_npz_name(AU_group_id, trainval_test, out_dir, database, fold, split_idx, sequence_key):
    if trainval_test == "trainval":
        file_name = out_dir + os.path.sep + "{0}_{1}_fold_{2}".format(database, fold,
                    split_idx) + "/train" + os.path.sep + sequence_key + "#{}.npz".format(AU_group_id)
    else:
        file_name = out_dir + os.path.sep + "{0}_{1}_fold_{2}".format(database, fold,
                    split_idx) + "/test" + os.path.sep + sequence_key + "#{}.npz".format(AU_group_id)
    return file_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='each batch size will be a new file')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='gpu that used to extract feature')
    parser.add_argument("--out_dir", '-o', default="/home/machen/dataset/extract_features/")
    parser.add_argument("--model", '-m', help="the AU R-CNN pretrained model file to load to extract feature")
    parser.add_argument("--trainval_test", '-tt', help="train or test")
    parser.add_argument("--database", default="BP4D")
    parser.add_argument('--use_memcached', action='store_true',
                        help='whether use memcached to boost speed of fetch crop&mask')
    parser.add_argument('--stack_frames', type=int, default=1)
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_no_enhance.npy",
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
    paper_report_label, class_num = squeeze_label_num_report(args.database, True)

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
    elif backbone == 'optical_flow':
        au_rcnn = AU_RCNN_Resnet101(pretrained_model="resnet101",
                                    min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                    mean_file=args.mean, classify_mode=False, n_class=class_num,
                                    use_roi_align=False, use_feature_map_res45=False,
                                    use_feature_map_res5=False,
                                    use_optical_flow_input=True,
                                    temporal_length=args.stack_frames)
        au_rcnn_train_chain = AU_RCNN_ROI_Extractor(au_rcnn)
        faster_rcnn = Wrapper([au_rcnn_train_chain], None, args.database, args.stack_frames, use_feature_map=False,
                    two_stream_mode=TwoStreamMode.optical_flow)


    assert os.path.exists(args.model)
    print("loading model file : {}".format(args.model))
    chainer.serializers.load_npz(args.model, faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        faster_rcnn.to_gpu(args.gpu)

    dataset = FeatureExtractorDataset(database=args.database,
                           fold=fold, split_name=args.trainval_test,
                           split_index=split_idx, mc_manager=mc_manager, use_lstm=False,
                           train_all_data=False,
                           prefix="", pretrained_target="", pretrained_model=faster_rcnn, extract_key="avg_pool",
                           device=-1, batch_size=args.batch_size
                           )
    dataset_iter = SerialIterator(dataset, batch_size=args.batch_size,
                                        repeat=False, shuffle=False)
    last_sequence_key = None
    features = []
    labels = []
    for batch in dataset_iter:

        for idx, (feature, label, sequence_key) in enumerate(batch):  # feature shape = R x 2048
            if last_sequence_key is None:
                last_sequence_key = sequence_key
            if sequence_key != last_sequence_key:  # 换video了
                if len(features) == 0:
                    print("all feature cannot obtain {}".format(last_sequence_key))
                features = np.stack(features)  # shape = N, R, 2048
                labels = np.stack(labels)  # shape = N, R, 12
                features_trans = np.transpose(features, axes=(1, 0, 2))  # shape = R, N, 2048
                labels_trans = np.transpose(labels, axes=(1,0,2))  # shape = R, N, 12

                for AU_group_id, box_feature in enumerate(features_trans):
                    output_filename = get_npz_name(AU_group_id + 1, args.trainval_test, args.out_dir, args.database, fold, split_idx, last_sequence_key)
                    print("write : {}".format(output_filename))
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    np.savez(output_filename, feature=features_trans[AU_group_id],  label=labels_trans[AU_group_id])
                features = []
                labels = []
                last_sequence_key = sequence_key

            if label is not None and label.shape[1] == config.BOX_NUM[args.database]:
                features.extend(feature)
                labels.extend(label)
            else:
                print("one batch cannot fetch from seq: {0}".format(sequence_key))


    # the last sequence
    features = np.stack(features)  # shape = N, R, 2048
    labels = np.stack(labels)  # shape = N, R, 12
    features_trans = np.transpose(features, axes=(1, 0, 2))  # shape = R, N, 2048
    labels_trans = np.transpose(labels, axes=(1, 0, 2))  # shape = R, N, 12

    for AU_group_id, box_feature in enumerate(features_trans):
        output_filename = get_npz_name(AU_group_id + 1, args.trainval_test, args.out_dir, args.database, fold,
                                       split_idx, last_sequence_key)
        print("write : {}".format(output_filename))
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        np.savez(output_filename, feature=features_trans[AU_group_id], label=labels_trans[AU_group_id])

if __name__ == "__main__":

    main()