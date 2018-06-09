import argparse
import sys

from chainer.dataset import concat_examples


sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
from time_axis_rcnn.extensions.special_converter import concat_examples_not_string
from time_axis_rcnn.model.time_segment_network.faster_head_module import FasterHeadModule
from time_axis_rcnn.model.time_segment_network.faster_rcnn_backbone import FasterBackbone
from time_axis_rcnn.model.time_segment_network.segment_proposal_network import SegmentProposalNetwork
from time_axis_rcnn.model.time_segment_network.tcn_backbone import TcnBackbone
from time_axis_rcnn.model.time_segment_network.wrapper_predictor import WrapperPredictor
from two_stream_rgb_flow.extensions.special_converter import concat_examples_not_labels

from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report

from chainer.iterators import SerialIterator
from time_axis_rcnn.model.time_segment_network.faster_rcnn_predictor import TimeSegmentRCNNPredictor
import chainer
from time_axis_rcnn.constants.enum_type import TwoStreamMode, FasterBackboneType

from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import os
from collections import OrderedDict
import json
import re
import config
from time_axis_rcnn.datasets.npz_feature_dataset import NpzFeatureDataset
from time_axis_rcnn.extensions.AU_evaluator import ActionUnitEvaluator

#  time_axis_rcnn_BP4D_3_fold_1@use_paper_num_label@rgb_flow@30_model.npz
def extract_mode(model_file_name):
    model_file_name = os.path.basename(model_file_name)
    pattern = re.compile('time_axis_rcnn_(.*?)_(.*?)_fold_(.*?)@(.*?)@(.*?)@(.*?)@(.*?)_model\.npz', re.DOTALL)
    matcher = pattern.match(model_file_name)
    return_dict = OrderedDict()
    if matcher:
        database = matcher.group(1)
        fold = int(matcher.group(2))
        split_idx = int(matcher.group(3))
        use_paper_num_label = True if matcher.group(4) == "use_paper_num_label" else False
        two_stream_mode = TwoStreamMode[matcher.group(5)]
        conv_layers = int(matcher.group(6))
        return_dict["database"] = database
        return_dict["fold"] = fold
        return_dict["split_idx"] = split_idx
        return_dict["use_paper_num_label"] = use_paper_num_label
        return_dict["two_stream_mode"] = two_stream_mode
        return_dict["conv_layers"] = conv_layers
        return_dict["faster_backbone_type"] = FasterBackboneType[matcher.group(7)]
    return return_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument("--model", "-m", help="pretrained model file path") # which contains pretrained target
    parser.add_argument('--proc_num', type=int, default=10, help="multiprocess fetch data process number")
    parser.add_argument("--data_dir", type=str, default="/home/machen/dataset/extract_features")
    parser.add_argument('--batch', '-b', type=int, default=1,
                        help='mini batch size')
    args = parser.parse_args()
    if not args.model.endswith("model.npz"):
        return

    mode_dict = extract_mode(args.model)


    database = mode_dict["database"]
    fold = mode_dict["fold"]
    split_idx = mode_dict["split_idx"]
    use_paper_num_label = mode_dict["use_paper_num_label"]
    conv_layers = mode_dict["conv_layers"]
    two_stream_mode = mode_dict["two_stream_mode"]
    faster_backbone_type = mode_dict["faster_backbone_type"]
    T = 10
    data_dir = args.data_dir + "/{0}_{1}_fold_{2}/test".format(database, fold, split_idx)

    adaptive_AU_database(database)
    paper_report_label, class_num = squeeze_label_num_report(database, use_paper_num_label)
    paper_report_label_idx = list(paper_report_label.keys())
    class_num = len(config.AU_SQUEEZE)
    if use_paper_num_label:
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
        """.format(args.model, json.dumps(model_print_dict, sort_keys=True, indent=8)))
    if faster_backbone_type == FasterBackboneType.conv1d:
        faster_extractor_backbone = FasterBackbone(conv_layers, 2048, 1024)
    elif faster_backbone_type == FasterBackboneType.tcn:
        faster_extractor_backbone = TcnBackbone(conv_layers, 2048, 1024)
    faster_head_module = FasterHeadModule(2048, class_num + 1, 7)  # note that the class number here must include background
    initialW = chainer.initializers.Normal(0.001)
    spn = SegmentProposalNetwork(1024, n_anchors=len(config.ANCHOR_SIZE), initialW=initialW)
    seg_predictor = TimeSegmentRCNNPredictor(faster_extractor_backbone, spn, faster_head_module)
    model = WrapperPredictor(seg_predictor, class_num=class_num)

    chainer.serializers.load_npz(args.model, model.seg_predictor.train_chain)
    print("loading {}".format(args.model))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    npz_dataset = NpzFeatureDataset(data_dir, database, two_stream_mode=two_stream_mode, T=T)

    test_iter = SerialIterator(npz_dataset, batch_size=1,
                                     repeat=False, shuffle=False)


    with chainer.no_backprop_mode(),chainer.using_config('cudnn_deterministic',True),chainer.using_config('train',False):
        # time_axis_rcnn_BP4D_3_fold_1@use_paper_num_label@rgb_flow@30_model.npz
        pred_result_npz_path = os.path.dirname(args.model) + os.path.sep + os.path.basename(args.model)[
                                                                    :os.path.basename(args.model).rindex("_")] + "_pred_result.npz"
        au_evaluator = ActionUnitEvaluator(test_iter, model, args.gpu, database=database,
                                           paper_report_label=paper_report_label,
                                           converter=lambda batch, device: concat_examples_not_string(batch, device, padding=0),
                                           output_path=pred_result_npz_path)
        observation = au_evaluator.evaluate()
        with open(os.path.dirname(args.model) + os.path.sep + "evaluation_result_{0}.json".format(os.path.basename(args.model)\
                                                                            [:os.path.basename(args.model).rindex("_")]
                                                           ), "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()


if __name__ == '__main__':
    main()