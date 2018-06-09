import sys
from collections import OrderedDict

from numpy import random

sys.path.insert(0, '/home/machen/face_expr')

from chainer.datasets import TransformDataset

from time_axis_rcnn.extensions.special_converter import concat_examples_not_string

from chainer.iterators import SerialIterator, MultiprocessIterator

from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report
from two_stream_rgb_flow import transforms
import argparse
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_resnet101 import FasterRCNNResnet101

from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_resnet101 import AU_RCNN_Resnet101
from two_stream_rgb_flow.model.wrap_model.wrapper import Wrapper
from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_train_chain import AU_RCNN_ROI_Extractor
from two_stream_rgb_flow.constants.enum_type import TwoStreamMode

from time_axis_rcnn.datasets.AU_dataset import AUDataset
import config
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from time_axis_rcnn.model.dump_feature_model.dump_feature_model import DumpRoIFeature

import re
import os
import chainer
import numpy as np


class Transform(object):

    def __init__(self, L, mean_rgb_path, mean_flow_path, mirror=True):
        self.mirror = mirror
        self.mean_rgb = np.load(mean_rgb_path)
        self.mean_rgb = self.mean_rgb.astype(np.float32)
        self.mean_flow = np.tile(np.expand_dims(np.load(mean_flow_path),  axis=0), reps=(L, 1, 1, 1))[:, :2, :, :]
        self.mean_flow = self.mean_flow.astype(np.float32)

    def __call__(self, in_data):
        rgb_img, flow_img_list, bbox, label, rgb_path = in_data  # flow_img_list shape = (T, C, H, W), and bbox = (F,4)
        if rgb_img is None:
            return None, None, None, None, rgb_path
        rgb_img = rgb_img - self.mean_rgb
        assert flow_img_list.shape == self.mean_flow.shape
        flow_imgs = flow_img_list - self.mean_flow

        assert len(np.where(bbox < 0)[0]) == 0
        # horizontally flip and random shift box
        if self.mirror:
            rgb_img = rgb_img[:, :, ::-1]
            flow_imgs = flow_imgs[:, :, :, ::-1] # (T, C, H, W) where W is flipped
            bbox = transforms.flip_bbox(
                bbox, (rgb_img.shape[0], rgb_img.shape[1]), x_flip=True)

        return rgb_img, flow_imgs, bbox, label, rgb_path



def extract_mode(model_file_name):
    model_file_name = os.path.basename(model_file_name)
    pattern = re.compile('(.*?)_(.*?)_fold_(.*?)_(.*?)@(.*?)@(.*?)@(.*?)@T#(.*?)_model\.npz', re.DOTALL)
    matcher = pattern.match(model_file_name)
    return_dict = OrderedDict()
    if matcher:
        database = matcher.group(1)
        fold = int(matcher.group(2))
        split_idx = int(matcher.group(3))
        backbone = matcher.group(4)
        two_stream_mode = TwoStreamMode[matcher.group(5)]
        use_paper_num_label = True if matcher.group(6) == "use_paper_num_label" else False
        roi_align = True if matcher.group(7) == "roi_align" else False
        T = int(matcher.group(8))

        return_dict["database"] = database
        return_dict["fold"] = fold
        return_dict["split_idx"] = split_idx
        return_dict["backbone"] = backbone
        return_dict["use_paper_num_label"] = use_paper_num_label
        return_dict["use_roi_align"] = roi_align
        return_dict["T"] = T
        return_dict["two_stream_mode"] = two_stream_mode

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
    parser.add_argument("--mirror", action="store_true", help="whether to mirror")
    parser.add_argument("--out_dir", '-o', default="/home/machen/dataset/extract_features/")
    parser.add_argument("--model", '-m', help="the AU R-CNN pretrained model file to load to extract feature")
    parser.add_argument("--trainval_test", '-tt', help="train or test")
    parser.add_argument("--database", default="BP4D")
    parser.add_argument('--use_memcached', action='store_true',
                        help='whether use memcached to boost speed of fetch crop&mask')
    parser.add_argument('--proc_num', type=int, default=10)
    parser.add_argument('--memcached_host', default='127.0.0.1')
    parser.add_argument('--mean_rgb', default=config.ROOT_PATH + "BP4D/idx/mean_rgb.npy", help='image mean .npy file')
    parser.add_argument('--mean_flow', default=config.ROOT_PATH + "BP4D/idx/mean_flow.npy", help='image mean .npy file')

    args = parser.parse_args()
    adaptive_AU_database(args.database)
    mc_manager = None
    if args.use_memcached:
        from collections_toolkit.memcached_manager import PyLibmcManager
        mc_manager = PyLibmcManager(args.memcached_host)
        if mc_manager is None:
            raise IOError("no memcached found listen in {}".format(args.memcached_host))

    return_dict = extract_mode(args.model)
    database = return_dict["database"]
    fold = return_dict["fold"]
    split_idx = return_dict["split_idx"]
    backbone = return_dict["backbone"]
    use_paper_num_label = return_dict["use_paper_num_label"]
    roi_align = return_dict["use_roi_align"]
    two_stream_mode = return_dict["two_stream_mode"]
    T = return_dict["T"]

    class_num = len(config.paper_use_BP4D) if database == "BP4D" else len(config.paper_use_DISFA)
    paper_report_label_idx = sorted(list(config.AU_SQUEEZE.keys()))
    if use_paper_num_label:
        paper_report_label, class_num = squeeze_label_num_report(database, True)
        paper_report_label_idx = list(paper_report_label.keys())

    assert two_stream_mode == TwoStreamMode.rgb_flow
    if two_stream_mode ==TwoStreamMode.rgb_flow:
        au_rcnn_train_chain_list = []
        au_rcnn_rgb = AU_RCNN_Resnet101(pretrained_model=backbone,
                                        min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                        use_roi_align=roi_align,
                                        use_optical_flow_input=False, temporal_length=T)

        au_rcnn_optical_flow = AU_RCNN_Resnet101(pretrained_model=backbone,
                                                 min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                                 use_roi_align=roi_align,
                                                 use_optical_flow_input=True, temporal_length=T)

        au_rcnn_train_chain_rgb = AU_RCNN_ROI_Extractor(au_rcnn_rgb)
        au_rcnn_train_chain_optical_flow = AU_RCNN_ROI_Extractor(au_rcnn_optical_flow)

        au_rcnn_train_chain_list.append(au_rcnn_train_chain_rgb)
        au_rcnn_train_chain_list.append(au_rcnn_train_chain_optical_flow)
        model = Wrapper(au_rcnn_train_chain_list, class_num, database, T,
                        two_stream_mode=two_stream_mode, gpus=[args.gpu, args.gpu])

    assert os.path.exists(args.model)
    print("loading model file : {}".format(args.model))
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        if isinstance(model, FasterRCNNResnet101):
            model.to_gpu(args.gpu)


    img_dataset = AUDataset(database=database, L=T,
                            fold=fold, split_name=args.trainval_test,
                            split_index=split_idx, mc_manager=mc_manager,
                            train_all_data=False,
                            paper_report_label_idx=paper_report_label_idx, jump_exists=True, npz_dir=args.out_dir)
    mirror_list = [False,]
    if args.mirror and args.trainval_test == 'trainval':
        mirror_list.append(True)
    for mirror in mirror_list:
        train_dataset = TransformDataset(img_dataset, Transform(T, mean_rgb_path=args.mean_rgb,
                                                                mean_flow_path=args.mean_flow, mirror=mirror))


        if args.proc_num > 1:
            dataset_iter = MultiprocessIterator(train_dataset, batch_size=args.batch_size,
                             n_processes=args.proc_num,
                             repeat=False, shuffle=False, n_prefetch=10, shared_mem=10000000)
        else:
            dataset_iter = SerialIterator(train_dataset, batch_size=args.batch_size,
                                          repeat=False, shuffle=False)


        with chainer.no_backprop_mode(), chainer.using_config('cudnn_deterministic', True), chainer.using_config(
            'train', False):
            model_dump = DumpRoIFeature(dataset_iter, model, args.gpu, database,
                                        converter=lambda batch, device: concat_examples_not_string(batch, device, padding=0),
                                        output_path=args.out_dir, trainval_test=args.trainval_test, fold_split_idx=split_idx, mirror_data=mirror)
            model_dump.evaluate()

if __name__ == "__main__":

    main()