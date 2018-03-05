import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from simple_graph_learning.extensions.AU_evaluator import ActionUnitEvaluator
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from simple_graph_learning.dataset.simple_feature_dataset import SimpleFeatureDataset
from simple_graph_learning.model.space_time_net.enum_type import SpatialEdgeMode, RecurrentType
import os
from collections import OrderedDict
from chainer.dataset import concat_examples
import json

from simple_graph_learning.model.space_time_net.space_time_rnn_type_2 import SpaceTimeRNN
import re
import config


def extract_mode(model_file_name):
    pattern = re.compile('.*?@(.*?)@(.*?)@(.*?)@(.*?)\.npz', re.DOTALL)
    matcher = pattern.match(model_file_name)
    return_dict = {}
    if matcher:
        database = matcher.group(1)
        use_paper_report_label_num = True if matcher.group(2) == "paper_AU_num_train" else False
        spatial_edge_mode = matcher.group(3)
        temporal_edge_mode = matcher.group(4)
        return_dict["database"] = database
        return_dict["spatial_edge_mode"] = spatial_edge_mode
        return_dict["temporal_edge_mode"] = temporal_edge_mode
        return_dict["use_paper_report_label_num"] = use_paper_report_label_num
    return return_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument("--model", "-m", help="pretrained model file path") # which contains pretrained target
    parser.add_argument("--test", "-tt", default="", help="test txt folder path")
    parser.add_argument("--database","-db",default="BP4D", help="which database you want to evaluate")
    parser.add_argument("--check", "-ck", action="store_true", help="default not to check the npy file and all list file generate correctly")
    parser.add_argument("--num_attrib",type=int,default=2048, help="feature dimension")
    parser.add_argument("--attn_heads",type=int, default=16)
    parser.add_argument("--layers", type=int, default=1, help="layer number of edge/node rnn")
    parser.add_argument("--bi_lstm", action="store_true", help="whether or not to use bi_lstm as edge/node rnn base")



    args = parser.parse_args()

    mode_dict = extract_mode(args.model)
    adaptive_AU_database(mode_dict['database'])
    paper_report_label = OrderedDict()
    if mode_dict["use_paper_report_label_num"]:
        for AU_idx, AU in sorted(config.AU_SQUEEZE.items(), key=lambda e: int(e[0])):
            if mode_dict['database'] == "BP4D":
                paper_use_AU = config.paper_use_BP4D
            elif mode_dict['database'] == "DISFA":
                paper_use_AU = config.paper_use_DISFA
            if AU in paper_use_AU:
                paper_report_label[AU_idx] = AU
    paper_report_label_idx = list(paper_report_label.keys())
    if not paper_report_label_idx:
        paper_report_label_idx = None
        class_num = len(config.AU_SQUEEZE)
    else:
        class_num = len(paper_report_label_idx)
    if RecurrentType[mode_dict["temporal_edge_mode"]] == RecurrentType.no_temporal:
        args.layers = 3

    test_dir = args.test if not args.test.endswith("/") else args.test[:-1]
    assert mode_dict['database'] in test_dir


    print("""
        ======================================
        gpu:{0}
        argument: 
                spatial_edge_mode:{1}
                temporal_edge_mode:{2}
                use_paper_report_label_num:{3}
        ======================================
        """.format(args.gpu,mode_dict["spatial_edge_mode"],
                   mode_dict["temporal_edge_mode"],
                   mode_dict["use_paper_report_label_num"]))


    model = SpaceTimeRNN(mode_dict['database'], args.layers, args.num_attrib, class_num, None,
                         spatial_edge_model=SpatialEdgeMode[mode_dict["spatial_edge_mode"]],
                     recurrent_block_type=RecurrentType[mode_dict["temporal_edge_mode"]], attn_heads=args.attn_heads,
                     bi_lstm=args.bi_lstm)

    print("loading {}".format(args.model))
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        dataset = SimpleFeatureDataset(directory=args.test, database=mode_dict['database'], each_file_pic_num=200,
                                       previous_frame=80,
                                       sample_pic_count=0, paper_report_label_idx=paper_report_label_idx,
                                       train_mode=False)
        test_iter = chainer.iterators.MultiprocessIterator(dataset, 100, shuffle=True, repeat=False,n_processes=5,n_prefetch=2,shared_mem=10000000)
        au_evaluator = ActionUnitEvaluator(test_iter, model, args.gpu, database=mode_dict['database'],
                                           paper_report_label=paper_report_label,
                                           converter=lambda batch, device: concat_examples(batch, device, padding=0))
        observation = au_evaluator.evaluate()
        print(observation)
        with open(os.path.dirname(args.model) + os.sep + "evaluation_result_{0}@{1}@{2}@{3}.json".format(mode_dict['database'],
                                                           SpatialEdgeMode[mode_dict["spatial_edge_mode"]],
                                                           RecurrentType[mode_dict["temporal_edge_mode"]],
                                                            mode_dict["use_paper_report_label_num"]
                                                           ), "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()


if __name__ == '__main__':
    main()