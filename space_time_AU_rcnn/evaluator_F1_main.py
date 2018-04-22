import argparse
import sys


sys.path = sys.path[1:]
sys.path.append("/home1/machen/face_expr")
from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report

from chainer.iterators import MultiprocessIterator, SerialIterator
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_resnet50 import AU_RCNN_Resnet50
from collections_toolkit.memcached_manager import PyLibmcManager
from space_time_AU_rcnn.extensions.AU_evaluator import ActionUnitEvaluator
from space_time_AU_rcnn.datasets.AU_dataset import AUDataset
from space_time_AU_rcnn.model.roi_space_time_net.label_dependency_rnn import LabelDependencyLayer
from space_time_AU_rcnn.model.roi_space_time_net.space_time_conv_lstm import SpaceTimeConv
from space_time_AU_rcnn.model.roi_space_time_net.space_time_fc_lstm import SpaceTimeLSTM
from space_time_AU_rcnn.model.roi_space_time_net.space_time_seperate_conv_lstm import SpaceTimeSepConv
from space_time_AU_rcnn.model.wrap_model.wrapper import Wrapper
from space_time_AU_rcnn import transforms
from chainer.datasets import TransformDataset
import chainer
from space_time_AU_rcnn.constants.enum_type import SpatialEdgeMode, TemporalEdgeMode, ConvRNNType

from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_resnet101 import AU_RCNN_Resnet101
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn_train_chain import AU_RCNN_ROI_Extractor, AU_RCNN_TrainChainLoss

from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import os
from collections import OrderedDict
import json
from space_time_AU_rcnn.datasets.AU_video_dataset import AU_video_dataset
import re
import config
import numpy as np
from space_time_AU_rcnn.extensions.special_converter import concat_examples_not_labels

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


# BP4D_3_fold_1_resnet101@rnn@no_temporal@use_paper_num_label@roi_align@label_dep_layer@convlstm@sampleframe#13_model.npz
def extract_mode(model_file_name):
    model_file_name = os.path.basename(model_file_name)
    pattern = re.compile('(.*?)_(.*?)_fold_(.*?)_(.*?)@(.*?)@(.*?)@(.*?)@(.*?)@(.*?)@(.*?)@sampleframe#(.*?)_model\.npz', re.DOTALL)
    matcher = pattern.match(model_file_name)
    return_dict = OrderedDict()
    if matcher:
        database = matcher.group(1)
        fold = int(matcher.group(2))
        split_idx = int(matcher.group(3))
        backbone = matcher.group(4)
        spatial_edge_mode = SpatialEdgeMode[matcher.group(5)]
        temporal_edge_mode = TemporalEdgeMode[matcher.group(6)]
        use_paper_num_label = True if matcher.group(7) == "use_paper_num_label" else False
        roi_align = True if matcher.group(8) == "roi_align" else False
        label_dep_rnn_layer = True if matcher.group(9) == "label_dep_layer" else False
        conv_rnn_type = ConvRNNType[matcher.group(10)]
        sample_frame = int(matcher.group(11))

        return_dict["database"] = database
        return_dict["fold"] = fold
        return_dict["split_idx"] = split_idx
        return_dict["backbone"] = backbone
        return_dict["spatial_edge_mode"] = spatial_edge_mode
        return_dict["temporal_edge_mode"] = temporal_edge_mode
        return_dict["use_paper_num_label"] = use_paper_num_label
        return_dict["use_roi_align"] = roi_align
        return_dict["label_dep_rnn_layer"] = label_dep_rnn_layer
        return_dict["conv_rnn_type"] = conv_rnn_type
        return_dict["sample_frame"] = sample_frame

    return return_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument("--model", "-m", help="pretrained model file path") # which contains pretrained target
    parser.add_argument("--pretrained_model", "-pre", default="resnet101")
    parser.add_argument("--memcached_host", default="127.0.0.1")
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_no_enhance.npy",
                        help='image mean .npy file')
    parser.add_argument('--proc_num', type=int, default=3, help="multiprocess fetch data process number")
    args = parser.parse_args()
    if not args.model.endswith("model.npz"):
        return
    mode_dict = extract_mode(args.model)
    database = mode_dict["database"]
    fold = mode_dict["fold"]
    split_idx = mode_dict["split_idx"]
    backbone = mode_dict["backbone"]
    spatial_edge_mode = mode_dict["spatial_edge_mode"]
    temporal_edge_mode = mode_dict["temporal_edge_mode"]
    use_paper_num_label = mode_dict["use_paper_num_label"]
    use_roi_align = mode_dict["use_roi_align"]
    use_label_dep_rnn_layer = mode_dict["label_dep_rnn_layer"]
    sample_frame = mode_dict["sample_frame"]
    conv_rnn_type = mode_dict["conv_rnn_type"]
    use_conv_lstm = (conv_rnn_type != ConvRNNType.conv_rcnn)
    adaptive_AU_database(database)
    paper_report_label, class_num = squeeze_label_num_report(database, use_paper_num_label)
    paper_report_label_idx = list(paper_report_label.keys())
    if not paper_report_label_idx:
        paper_report_label_idx = None
        class_num = len(config.AU_SQUEEZE)
    else:
        class_num = len(paper_report_label_idx)

    model_print_dict = OrderedDict()
    for key, value in mode_dict.items():
        model_print_dict[key] = str(value)
    print("""
        {0}
        ======================================
        INFO:
        {1}
        ======================================
        """.format(args.model,json.dumps(model_print_dict, sort_keys=True, indent=8)))
    if backbone == 'resnet101':
        au_rcnn = AU_RCNN_Resnet101(pretrained_model=args.pretrained_model,
                                    min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                    mean_file=args.mean, classify_mode=(not use_conv_lstm), n_class=class_num,
                                    use_roi_align=use_roi_align, use_feature_map=use_conv_lstm,
                                    use_feature_map_res5=(
                                    conv_rnn_type == ConvRNNType.fc_lstm or conv_rnn_type == ConvRNNType.sep_conv_lstm))
    elif backbone == 'resnet50':
        au_rcnn = AU_RCNN_Resnet50(pretrained_model=args.pretrained_model,min_size=config.IMG_SIZE[0],
                                   max_size=config.IMG_SIZE[1], mean_file=args.mean,classify_mode=(not use_conv_lstm),
                                   n_class=class_num,  use_roi_align=use_roi_align, use_feature_map=use_conv_lstm
                                   )
    au_rcnn_train_chain = AU_RCNN_ROI_Extractor(au_rcnn)


    # if use_label_dep_rnn_layer:
    #     use_space = (spatial_edge_mode != SpatialEdgeMode.no_edge)
    #     use_temporal = (temporal_edge_mode != TemporalEdgeMode.no_temporal)
    #     label_dependency_layer = LabelDependencyLayer(database, out_size=class_num, train_mode=False,
    #                                                   label_win_size=2, x_win_size=1,
    #                                                   label_dropout_ratio=0.0, use_space=use_space,
    #                                                   use_temporal=use_temporal)
    if conv_rnn_type == ConvRNNType.conv_lstm:
        space_time_conv_lstm = SpaceTimeConv(None, use_label_dep_rnn_layer, class_num,
                                             spatial_edge_mode=spatial_edge_mode, temporal_edge_mode=temporal_edge_mode,
                                             conv_rnn_type=conv_rnn_type)
        loss_head_module = space_time_conv_lstm
    elif conv_rnn_type == ConvRNNType.fc_lstm:
        space_time_fc_lstm =SpaceTimeLSTM(class_num, spatial_edge_mode=spatial_edge_mode, temporal_edge_mode=temporal_edge_mode)
        loss_head_module = space_time_fc_lstm
    elif conv_rnn_type == ConvRNNType.conv_rcnn:
        au_rcnn_train_loss = AU_RCNN_TrainChainLoss()
        loss_head_module = au_rcnn_train_loss
    elif conv_rnn_type == ConvRNNType.sep_conv_lstm:
        space_time_sep_conv_lstm = SpaceTimeSepConv(database, class_num, spatial_edge_mode=spatial_edge_mode, temporal_edge_mode=temporal_edge_mode)
        loss_head_module = space_time_sep_conv_lstm

    model = Wrapper(au_rcnn_train_chain, loss_head_module, database, sample_frame,
                        use_feature_map=use_conv_lstm)
    chainer.serializers.load_npz(args.model, model)
    print("loading {}".format(args.model))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    mc_manager = PyLibmcManager(args.memcached_host)
    img_dataset = AUDataset(database=database,
                            fold=fold, split_name='test',   # FIXME
                            split_index=split_idx, mc_manager=mc_manager,
                            train_all_data=False)

    video_dataset = AU_video_dataset(au_image_dataset=img_dataset, sample_frame=sample_frame, train_mode=False,  #FIXME
                    paper_report_label_idx=paper_report_label_idx,fetch_use_parrallel_iterator=True)

    video_dataset = TransformDataset(video_dataset, Transform3D(au_rcnn, mirror=False))

    # test_iter = SerialIterator(video_dataset, batch_size=sample_frame,
    #                                  repeat=False, shuffle=False, )

    test_iter = MultiprocessIterator(video_dataset, batch_size=sample_frame,
                                       n_processes=args.proc_num,
                                       repeat=False, shuffle=False, n_prefetch=10, shared_mem=314572800)



    with chainer.no_backprop_mode(), chainer.using_config('train', False):

        au_evaluator = ActionUnitEvaluator(test_iter, model, args.gpu, database=database,
                                           paper_report_label=paper_report_label,
                                           converter=lambda batch, device: concat_examples_not_labels(batch, device, padding=0),
                                           use_feature_map=use_conv_lstm, sample_frame=sample_frame)
        observation = au_evaluator.evaluate()
        print(observation)
        with open(os.path.dirname(args.model) + os.sep + "evaluation_result_{0}.json".format(os.path.basename(args.model)\
                                                                            [:os.path.basename(args.model).rindex("_")]
                                                           ), "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()


if __name__ == '__main__':
    main()