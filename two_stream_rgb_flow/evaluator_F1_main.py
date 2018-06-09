import argparse
import sys


sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
from dataset_toolkit.squeeze_label_num_report import squeeze_label_num_report

from chainer.iterators import MultiprocessIterator, SerialIterator
from two_stream_rgb_flow.datasets.AU_dataset import AUDataset
from collections_toolkit.memcached_manager import PyLibmcManager
from two_stream_rgb_flow.extensions.AU_evaluator import ActionUnitEvaluator
from two_stream_rgb_flow.model.wrap_model.wrapper import Wrapper
from two_stream_rgb_flow import transforms
from two_stream_rgb_flow.constants.enum_type import TwoStreamMode
from chainer.datasets import TransformDataset
import chainer

from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_resnet101 import AU_RCNN_Resnet101
from two_stream_rgb_flow.model.AU_rcnn.au_rcnn_train_chain import AU_RCNN_ROI_Extractor

from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import os
from collections import OrderedDict
import json
import re
import config
import numpy as np
from two_stream_rgb_flow.extensions.special_converter import concat_examples_not_labels

class Transform(object):

    def __init__(self, L, mean_rgb_path, mean_flow_path):
        self.mean_rgb = np.load(mean_rgb_path)
        self.mean_rgb = self.mean_rgb.astype(np.float32)
        self.mean_flow = np.tile(np.expand_dims(np.load(mean_flow_path),  axis=0), reps=(L, 1, 1, 1))[:, :2, :, :]
        self.mean_flow = self.mean_flow.astype(np.float32)

    def __call__(self, in_data):
        rgb_img, flow_img_list, bbox, label = in_data  # flow_img_list shape = (T, C, H, W), and bbox = (F,4)
        rgb_img = rgb_img - self.mean_rgb
        assert flow_img_list.shape == self.mean_flow.shape
        flow_img = flow_img_list - self.mean_flow

        assert len(np.where(bbox < 0)[0]) == 0
        # horizontally flip and random shift box
        return rgb_img, flow_img, bbox, label


# BP4D_3_fold_1_resnet101@rgb_flow@use_paper_num_label@roi_pooling@T#10_model.npz
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument("--model", "-m", help="pretrained model file path") # which contains pretrained target
    parser.add_argument("--pretrained_model", "-pre", default="resnet101")
    parser.add_argument("--memcached_host", default="127.0.0.1")
    parser.add_argument('--mean_rgb', default=config.ROOT_PATH + "BP4D/idx/mean_rgb.npy", help='image mean .npy file')
    parser.add_argument('--mean_flow', default=config.ROOT_PATH + "BP4D/idx/mean_flow.npy", help='image mean .npy file')
    parser.add_argument('--proc_num', type=int, default=10, help="multiprocess fetch data process number")
    parser.add_argument('--batch', '-b', type=int, default=10,
                        help='mini batch size')
    args = parser.parse_args()
    if not args.model.endswith("model.npz"):
        return
    model_info = extract_mode(args.model)
    database = model_info["database"]
    fold = model_info["fold"]
    split_idx = model_info["split_idx"]
    backbone = model_info["backbone"]
    use_paper_num_label = model_info["use_paper_num_label"]
    use_roi_align = model_info["use_roi_align"]
    two_stream_mode = model_info['two_stream_mode']
    T = model_info["T"]

    adaptive_AU_database(database)
    paper_report_label, class_num = squeeze_label_num_report(database, use_paper_num_label)
    paper_report_label_idx = list(paper_report_label.keys())
    if not paper_report_label_idx:
        paper_report_label_idx = None
        class_num = len(config.AU_SQUEEZE)
    else:
        class_num = len(paper_report_label_idx)

    model_print_dict = OrderedDict()
    for key, value in model_info.items():
        model_print_dict[key] = str(value)
    print("""
        {0}
        ======================================
        INFO:
        {1}
        ======================================
        """.format(args.model,json.dumps(model_print_dict, sort_keys=True, indent=8)))

    au_rcnn_train_chain_list = []
    if backbone == 'resnet101':
        if two_stream_mode != TwoStreamMode.rgb_flow:
            pretrained_model = backbone
            au_rcnn = AU_RCNN_Resnet101(pretrained_model=pretrained_model,
                                        min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                        use_roi_align=use_roi_align,
                                        use_optical_flow_input=(two_stream_mode == TwoStreamMode.optical_flow),
                                        temporal_length=T)
            au_rcnn_train_chain = AU_RCNN_ROI_Extractor(au_rcnn)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain)
        else: # rgb_flow mode
            au_rcnn_rgb = AU_RCNN_Resnet101(pretrained_model=backbone,
                                            min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                            use_roi_align=use_roi_align,
                                            use_optical_flow_input=False, temporal_length=T)


            au_rcnn_optical_flow = AU_RCNN_Resnet101(pretrained_model=backbone,
                                                     min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[1],
                                                     use_roi_align=use_roi_align,
                                                     use_optical_flow_input=True, temporal_length=T)
            au_rcnn_train_chain_rgb = AU_RCNN_ROI_Extractor(au_rcnn_rgb)
            au_rcnn_train_chain_optical_flow = AU_RCNN_ROI_Extractor(au_rcnn_optical_flow)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_rgb)
            au_rcnn_train_chain_list.append(au_rcnn_train_chain_optical_flow)
            au_rcnn = au_rcnn_rgb

    model = Wrapper(au_rcnn_train_chain_list, class_num, database, T,
                    two_stream_mode=two_stream_mode, gpus=[args.gpu, args.gpu])

    chainer.serializers.load_npz(args.model, model)
    print("loading {}".format(args.model))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    mc_manager = PyLibmcManager(args.memcached_host)
    img_dataset = AUDataset(database=database, L=T,
                            fold=fold, split_name='test',
                            split_index=split_idx, mc_manager=mc_manager,
                            train_all_data=False, two_stream_mode=two_stream_mode, paper_report_label_idx=paper_report_label_idx)

    video_dataset = TransformDataset(img_dataset, Transform(L=T,mean_rgb_path=args.mean_rgb,
                                                                    mean_flow_path=args.mean_flow))
    if args.proc_num == 1:
        test_iter = SerialIterator(video_dataset, batch_size=args.batch, repeat=False, shuffle=False)
    else:
        test_iter = MultiprocessIterator(video_dataset, batch_size=args.batch,
                                       n_processes=args.proc_num,
                                       repeat=False, shuffle=False, n_prefetch=10, shared_mem=10000000)


    with chainer.no_backprop_mode(),chainer.using_config('cudnn_deterministic',True),chainer.using_config('train',False):
        predict_data_path = os.path.dirname(args.model) + os.path.sep + "pred_" + os.path.basename(args.model)[:os.path.basename(args.model).rindex("_")] + ".npz"
        print("npz_path: {}".format(predict_data_path))
        au_evaluator = ActionUnitEvaluator(test_iter, model, args.gpu, database=database,
                                           paper_report_label=paper_report_label,
                                           converter=lambda batch, device: concat_examples_not_labels(batch, device, padding=0),
                                           T=T, output_path=predict_data_path)
        observation = au_evaluator.evaluate()
        with open(os.path.dirname(args.model) + os.path.sep + "evaluation_result_{0}.json".format(os.path.basename(args.model)\
                                                                            [:os.path.basename(args.model).rindex("_")]
                                                           ), "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()


if __name__ == '__main__':
    main()