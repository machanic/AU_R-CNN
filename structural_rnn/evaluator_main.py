import argparse
import sys
sys.path = sys.path[1:]
sys.path.append("/home/machen/face_expr")
import chainer
from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
from structural_rnn.dataset.structural_RNN_dataset import S_RNNPlusDataset
from structural_rnn.extensions.AU_evaluator_respectively import ActionUnitEvaluator # FIXME
from structural_rnn.dataset.crf_pact_structure import CRFPackageStructure
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import os

import json
from structural_rnn.model.s_rnn.s_rnn_plus import StructuralRNNPlus
from structural_rnn.model.open_crf.cython.open_crf_layer import OpenCRFLayer

import re

def check_pretrained_model_match_file(target_dict, test_folder): # where we should place train and test in this folder, this folder also contains pretrained npz file
    npy_file_list = []
    for folder in os.listdir(test_folder):
        if os.path.isfile(test_folder + os.sep + folder):
            npy_file = folder
            npy_file_list.append(npy_file[:npy_file.rindex(".")])
    for folder in os.listdir(test_folder):
        if os.path.isdir(test_folder + os.sep + folder):
            if folder not in target_dict:
                return False
            for sub_folder in os.listdir(test_folder + os.sep + folder):
                if os.path.isdir(test_folder + os.sep + folder + os.sep + sub_folder):
                    for txt_file_path in os.listdir(test_folder + os.sep + folder + os.sep + sub_folder):
                        txt_file_path = txt_file_path[:txt_file_path.rindex(".")]
                        if txt_file_path not in npy_file_list:
                            return False
    return True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')  # open_crf layer only works for CPU mode
    parser.add_argument("--target_dir", "-t", default="result", help="pretrained model file path") # which contains pretrained target
    parser.add_argument("--test", "-tt", default="", help="test txt folder path")
    parser.add_argument("--hidden_size", "-hs",default=1024, type=int, help="hidden_size of srnn++")
    parser.add_argument("--database","-db",default="BP4D", help="which database you want to evaluate")
    parser.add_argument("--bi_lstm","-bi", action="store_true", help="srnn++ use bi_lstm or not, if pretrained model use bi_lstm, you must set this flag on")
    parser.add_argument("--check", "-ck", action="store_true", help="default not to check the npy file and all list file generate correctly")
    parser.add_argument("--num_attrib",type=int,default=2048, help="feature dimension")
    parser.add_argument("--train_edge",default="all",help="all/spatio/temporal")
    args = parser.parse_args()
    adaptive_AU_database(args.database)
    test_dir = args.test if not args.test.endswith("/") else args.test[:-1]
    assert args.database in test_dir
    dataset = GlobalDataSet(num_attrib=args.num_attrib, train_edge=args.train_edge) # ../data_info.json
    file_name = None
    for folder in os.listdir(args.test):
        if os.path.isdir(args.test + os.sep + folder):
            for _file_name in os.listdir(args.test + os.sep + folder):
                file_name = args.test + os.sep + folder  + os.sep +_file_name
                break
            break
    sample = dataset.load_data(file_name)
    print("pre load done")


    target_dict = {}
    need_srnn = False
    use_crf = False
    for model_path in os.listdir(args.target_dir):  # all model pretrained file in 3_fold_1's one folder, 3_fold_2 in another folder
        if model_path.endswith("model.npz"):
            assert ("opencrf" in model_path or "srnn_plus" in model_path)
            if "opencrf" in model_path:
                assert need_srnn == False
                use_crf = True
                # note that open_crf layer doesn't support GPU
                crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=dataset.num_attrib_type, need_s_rnn=False)
                model = OpenCRFLayer(node_in_size=dataset.num_attrib_type, weight_len=crf_pact_structure.num_feature)
                print("loading {}".format(args.target_dir + os.sep + model_path, model))
                chainer.serializers.load_npz(args.target_dir + os.sep + model_path, model)
            elif "srnn_plus" in model_path:
                crf_pact_structure = CRFPackageStructure(sample, dataset, num_attrib=args.hidden_size, need_s_rnn=True)
                with_crf = "crf" in model_path
                need_srnn = True
                model = StructuralRNNPlus(crf_pact_structure, in_size=dataset.num_attrib_type,
                                          out_size=dataset.num_label,
                                          hidden_size=args.hidden_size, with_crf=with_crf,
                                          use_bi_lstm=args.bi_lstm)  # if you train bi_lstm model in pretrained model, this time you need to use bi_lstm = True
                print("loading {}".format(args.target_dir + os.sep + model_path))
                chainer.serializers.load_npz(args.target_dir + os.sep + model_path, model)
                if args.gpu >= 0:
                    chainer.cuda.get_device_from_id(args.gpu).use()
                    model.to_gpu(args.gpu)
                    if with_crf:
                        model.open_crf.to_cpu()
            trainer_keyword_pattern = re.compile(".*?((\d+_)+)_*")
            matcher = trainer_keyword_pattern.match(model_path)
            assert matcher
            trainer_keyword = matcher.group(1)[:-1]
            target_dict[trainer_keyword] = model
    if len(target_dict) == 0:
        print("error , no pretrained npz file in {}".format(args.target_dir))
        return
    if args.check:
        check_pretrained_model_match_file(target_dict, args.test)
    with chainer.no_backprop_mode():
        test_data = S_RNNPlusDataset(directory=args.test, attrib_size=args.hidden_size, global_dataset=dataset,
                                     need_s_rnn=need_srnn, need_cache_factor_graph=False, target_dict=target_dict)  # if there is one file that use s_rnn, all the pact_structure need s_rnn
        test_iter = chainer.iterators.SerialIterator(test_data, 1, shuffle=False, repeat=False)
        gpu = args.gpu if not use_crf else -1
        print('using gpu :{}'.format(gpu))
        chainer.config.train = False
        with chainer.no_backprop_mode():
            au_evaluator = ActionUnitEvaluator(test_iter, target_dict, device=gpu, database=args.database)
            observation = au_evaluator.evaluate()
        with open(args.target_dir + os.sep + "evaluation_result.json", "w") as file_obj:
            file_obj.write(json.dumps(observation, indent=4, separators=(',', ': ')))
            file_obj.flush()

if __name__ == '__main__':
    main()