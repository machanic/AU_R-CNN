import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
from structural_rnn.dataset.structural_RNN_dataset import S_RNNPlusDataset
from structural_rnn.extensions.AU_evaluator import ActionUnitEvaluator
from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
import os

import json
from structural_rnn.model.s_rnn.s_rnn_plus import StructuralRNNPlus
from structural_rnn.model.open_crf.cython.open_crf_layer import OpenCRFLayer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument("--target_dir", "-t", default="result", help="pretrained model file path")
    parser.add_argument("--test", "-tt", default="", help="test txt folder path")
    parser.add_argument("--hidden_size", "-hs",default=1024, type=int, help="hidden_size of srnn++")
    parser.add_argument("--use_bi_lstm","-bi", action="store_true", help="srnn++ use bi_lstm or not")
    args = parser.parse_args()
    test_dir = args.test if not args.test.endswith("/") else args.test[:-1]
    dataset = GlobalDataSet(os.path.dirname(test_dir) + os.sep + "data_info.json")
    file_name = None
    for folder in os.listdir(args.test):
        if os.path.isdir(args.test + os.sep + folder):
            for _file_name in os.listdir(args.test + os.sep + folder):
                file_name = args.test + os.sep + folder  + os.sep +_file_name
                break
            break
    sample = dataset.load_data(file_name)
    print("pre load done")
    crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=dataset.num_attrib_type, need_s_rnn=True)
    target_dict = {}
    need_srnn = False
    for model_path in os.listdir(args.target_dir):  # 3_fold_1 in one folder, 3_fold_2 in another folder
        if model_path.endswith("model.npz"):
            assert ("opencrf" in model_path or "srnn_plus" in model_path)
            if "opencrf" in model_path:
                model = OpenCRFLayer(node_in_size=dataset.num_attrib_type, weight_len=crf_pact_structure.num_feature)
                chainer.serializers.load_npz(model_path, model)
            elif "srnn_plus" in model_path:
                with_crf = "crf" in model_path
                need_srnn = True
                model = StructuralRNNPlus(crf_pact_structure, in_size=dataset.num_attrib_type,
                                          out_size=dataset.label_bin_len,
                                          hidden_size=args.hidden_size, with_crf=with_crf,
                                          use_bi_lstm=args.use_bi_lstm)
                chainer.serializers.load_npz(model_path, model)
            train_keyword = os.path.basename(model_path)[:os.path.basename(model_path).index("_")]
            target_dict[train_keyword] = model
    with chainer.no_backprop_mode():
        test_data = S_RNNPlusDataset(directory=args.test, attrib_size=dataset.num_attrib_type, global_dataset=dataset,
                                     need_s_rnn=need_srnn, need_cache_factor_graph=True)
        test_iter = chainer.iterators.SerialIterator(test_data, 1, shuffle=False)
        au_evaluator = ActionUnitEvaluator(test_iter, model, device=-1, database=args.database)
        observation = au_evaluator.evaluate()
        with open(args.target_dir + os.sep + "evaluation_result.json", "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()

if __name__ == '__main__':
    main()