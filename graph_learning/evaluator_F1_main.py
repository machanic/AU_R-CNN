import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from graph_learning.dataset.graph_dataset_reader import GlobalDataSet
from graph_learning.dataset.graph_dataset import GraphDataset
from graph_learning.extensions.AU_evaluator import ActionUnitEvaluator # FIXME
from graph_learning.dataset.crf_pact_structure import CRFPackageStructure
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import os
from collections import OrderedDict

import json
from graph_learning.model.st_attention_net.st_attention_net_plus import StAttentioNetPlus
from graph_learning.model.open_crf.cython.open_crf_layer import OpenCRFLayer
from graph_learning.model.st_attention_net.enum_type import RecurrentType, NeighborMode, SpatialEdgeMode
from graph_learning.model.st_attention_net.st_relation_net_plus import StRelationNetPlus
import re
import config


def extract_mode(model_file_name):
    pattern = re.compile('.*?@(.*?)@.*?@(.*?)@(.*?)@(.*?)@(.*?)\.npz', re.DOTALL)
    matcher = pattern.match(model_file_name)
    return_dict = {}
    if matcher:
        use_paper_report_label_num = True if matcher.group(1) == "paper_AU_num_train" else False
        neighbor_mode = matcher.group(2)
        spatial_edge_mode = matcher.group(3)
        temporal_edge_mode = matcher.group(4)
        use_geo_feature = True if matcher.group(5) == "use_geo" else False
        return_dict["neighbor_mode"] = neighbor_mode
        return_dict["spatial_edge_mode"] = spatial_edge_mode
        return_dict["temporal_edge_mode"] = temporal_edge_mode
        return_dict["use_geo_feature"] = use_geo_feature
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
    parser.add_argument("--geo_num_attrib", type=int, default=4, help='geometry feature dimension')
    parser.add_argument("--train_edge",default="all",help="all/spatio/temporal")
    parser.add_argument("--attn_heads",type=int, default=16)
    parser.add_argument("--layers", type=int, default=1, help="layer number of edge/node rnn")
    parser.add_argument("--bi_lstm", action="store_true", help="whether or not to use bi_lstm as edge/node rnn base")
    parser.add_argument("--use_relation_net", action='store_true',
                        help='whether to use st_relation_net instead of st_attention_net')
    parser.add_argument("--relation_net_lstm_first", action='store_true',
                        help='whether to use relation_net_lstm_first_forward in st_relation_net')



    args = parser.parse_args()
    adaptive_AU_database(args.database)
    mode_dict = extract_mode(args.model)

    paper_report_label = OrderedDict()
    if mode_dict["use_paper_report_label_num"]:
        for AU_idx, AU in sorted(config.AU_SQUEEZE.items(), key=lambda e: int(e[0])):
            if args.database == "BP4D":
                paper_use_AU = config.paper_use_BP4D
            elif args.database == "DISFA":
                paper_use_AU = config.paper_use_DISFA
            if AU in paper_use_AU:
                paper_report_label[AU_idx] = AU
    paper_report_label_idx = list(paper_report_label.keys())
    if not paper_report_label_idx:
        paper_report_label_idx = None

    test_dir = args.test if not args.test.endswith("/") else args.test[:-1]
    assert args.database in test_dir
    dataset = GlobalDataSet(num_attrib=args.num_attrib, num_geo_attrib=args.geo_num_attrib, train_edge=args.train_edge) # ../data_info.json
    file_name = None
    for _file_name in os.listdir(args.test):
        if os.path.exists(args.test + os.sep + _file_name) and _file_name.endswith(".txt"):
            file_name = args.test + os.sep + _file_name
            break
    sample = dataset.load_data(file_name, npy_in_parent_dir=False, paper_use_label_idx=paper_report_label_idx)
    print("pre load done")

    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=dataset.num_attrib_type, need_s_rnn=False)
    print("""
        ======================================
        gpu:{4}
        argument: 
                neighbor_mode:{0}
                spatial_edge_mode:{1}
                temporal_edge_mode:{2}
                use_geometry_features:{3}
                use_paper_report_label_num:{5}
        ======================================
        """.format(mode_dict["neighbor_mode"], mode_dict["spatial_edge_mode"],
                   mode_dict["temporal_edge_mode"], mode_dict["use_geo_feature"], args.gpu,
                   mode_dict["use_paper_report_label_num"]))
    if args.use_relation_net:
        model = StRelationNetPlus(crf_pact_structure, in_size=dataset.num_attrib_type, out_size=dataset.label_bin_len,
                                  database=args.database, neighbor_mode=NeighborMode[mode_dict["neighbor_mode"]],
                                  spatial_edge_mode=SpatialEdgeMode[mode_dict["spatial_edge_mode"]],
                                  recurrent_block_type=RecurrentType[mode_dict["temporal_edge_mode"]],
                                  attn_heads=args.attn_heads, dropout=0.0,
                                  use_geometry_features=mode_dict["use_geo_feature"],
                                  layers=args.layers, bi_lstm=args.bi_lstm,
                                  lstm_first_forward=args.relation_net_lstm_first)
    else:
        model = StAttentioNetPlus(crf_pact_structure, dataset.num_attrib_type, dataset.label_bin_len,
                              args.database, NeighborMode[mode_dict["neighbor_mode"]], SpatialEdgeMode[mode_dict["spatial_edge_mode"]],
                              RecurrentType[mode_dict["temporal_edge_mode"]], attn_heads=args.attn_heads, dropout=0.0,
                              use_geometry_features=mode_dict["use_geo_feature"], layers=args.layers, bi_lstm=args.bi_lstm)
    print("loading {}".format(args.model))
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        test_data = GraphDataset(directory=test_dir, attrib_size=dataset.num_attrib_type, global_dataset=dataset,
                                 need_s_rnn=True, npy_in_parent_dir=False,
                                 need_cache_factor_graph=False, get_geometry_feature=True,
                                 paper_use_label_idx=paper_report_label_idx)
        test_iter = chainer.iterators.SerialIterator(test_data, 1, shuffle=False, repeat=False)
        au_evaluator = ActionUnitEvaluator(test_iter, model, args.gpu, database=args.database,
                                           paper_report_label=paper_report_label)
        observation = au_evaluator.evaluate()
        with open(os.path.dirname(args.model) + os.sep + "evaluation_result_{0}@{1}@{2}@{3}@{4}.json".format(args.database,
                                                           NeighborMode[mode_dict["neighbor_mode"]],
                                                           SpatialEdgeMode[mode_dict["spatial_edge_mode"]],
                                                           RecurrentType[mode_dict["temporal_edge_mode"]],
                                                           mode_dict["use_geo_feature"]), "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()


if __name__ == '__main__':
    main()