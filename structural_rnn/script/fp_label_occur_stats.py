import sys
from collections import defaultdict
import bisect
from optparse import OptionParser
import os
import itertools
import sys
sys.path.append("/home/machen/face_expr")
import random

def read_BP4D_AU_file(file_path):
    line_ls = []

    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.rstrip()
            terms = line.split()[1].split(",")
            line_ls.append(terms)
    print("data read over! data's count is:%s" % (len(line_ls)))
    return line_ls



if __name__ == "__main__":

    MSG_USAGE = """
    warning: only python2.6/2.7 is tested
    USAGE:
    python fp_tree.py --data_path ./test_data.txt --min_support 3 --output ./out.log OR --output stdout
    OR:
    python fp_tree.py -d ./test_data.txt -m 3 -o ./out.log OR -o stdout
    """
    opt = OptionParser(usage=MSG_USAGE, version="fp_tree version 1.0")
    opt.add_option("-d", "--data_path", type='string',
                   dest='data_path', help="data_path must required")
    opt.add_option("-m", "--min_support", type='int',
                   dest='min_support', help="min_support[int type] must required")
    opt.add_option("-o", "--output", type='string', dest='output', default="stdout",
                   help="output[default stdout]")
    opt.add_option( "--database", type='string', default="BP4D",
                   help="output[default stdout]")
    options, args = opt.parse_args()
    data_base = options.database
    data_path = options.data_path
    if data_path is None:
        opt.error("error, data_path not given!")
    min_support = options.min_support
    if min_support is None:
        opt.error("error, min_support not given!")
    if data_base == "BP4D":
        data = read_BP4D_AU_file(data_path)
    elif data_base == "DISFA":
        data =read_BP4D_AU_file(data_path)
    import pyfpgrowth
    from collections import defaultdict
    import config
    from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
    au_couple_dict = get_zip_ROI_AU()
    roi_set = set(list(au_couple_dict.values()))
    AU_belong_roi_id  = {"0":999}

    for idx, couple in enumerate(list(set(au_couple_dict.values()))):
        for AU in couple:
            AU_belong_roi_id[AU] = idx
    DISFA_roi_AU = defaultdict(list)
    if data_base == "DISFA":
        all_print_tuple = set()
        for AU in config.DISFA_use_AU:
            DISFA_roi_AU[AU_belong_roi_id[AU]].append(AU)
        for entry in itertools.product(*DISFA_roi_AU.values()):
            print_tuple = list(map(str, sorted(map(int, list(entry)))))
            while len(print_tuple) < config.BOX_NUM[data_base]:
                AU_1 = random.choice(["1","2","5"])
                if AU_1 not in print_tuple:
                    print_tuple.append(AU_1)

            print_tuple = list(map(str, sorted(map(int, list(print_tuple)))))
            assert len(print_tuple) == config.BOX_NUM[data_base]
            all_print_tuple.add(tuple(print_tuple))

        for entry in all_print_tuple:
            print(entry)
    elif data_base == "BP4D":
        freq_seqs = pyfpgrowth.find_frequent_patterns(data, 10000)
        rules = pyfpgrowth.generate_association_rules(freq_seqs, 0.6)   # {('12', '24'): (('17',), 0.841812865497076), ...}
        defined_rules = {}
        for AU_couple, rule in rules.items():
            defined_rules[tuple(sorted(AU_couple,key=lambda e:int(e)))] = rule[0][0]
        # output result frequent items
        all_print_tuple = set()

        for tu in sorted(freq_seqs.items(), key=lambda e:e[1], reverse=True):  # tu = (("10","12"), 23430) AU_couple, count
            roi_dict = defaultdict(list)
            for AU in tu[0]:
                roi_dict[AU_belong_roi_id[AU]].append(AU)
            if len(roi_dict) < 3:  # delete < 3 box couple
                continue
            for entry in itertools.product(*roi_dict.values()): # each roi choose one AU
                print_tuple = list(map(str, sorted(map(int, list(entry)))))
                all_print_tuple.add(tuple(print_tuple))

        final_tuple_list = set()
        for print_tuple in all_print_tuple:

            if len(print_tuple) > 2:

                already_use_roi = set()
                for AU in print_tuple:
                    already_use_roi.add(AU_belong_roi_id[AU])
                while len(print_tuple) < config.BOX_NUM[data_base]:
                    print_tuple = list(print_tuple)
                    roi_AU = defaultdict(list)
                    for AU_ in config.BP4D_use_AU:
                        roi_AU[AU_belong_roi_id[AU_]].append(AU_)
                    for i in range(2, len(print_tuple)+1):
                        for couple in itertools.combinations(print_tuple, i):
                            couple = tuple(sorted(couple,key=lambda e:int(e)))
                            if couple in defined_rules:
                                print("add rule")
                                rule_AU = defined_rules[couple]
                                if rule_AU not in print_tuple and AU_belong_roi_id[rule_AU] not in already_use_roi:
                                    print_tuple.append(rule_AU)
                                    already_use_roi.add(AU_belong_roi_id[rule_AU])
                    # random choice part
                    for roi_id in already_use_roi:
                        if roi_id in roi_AU:
                            del roi_AU[roi_id]
                    AU_choice = random.choice(["1","2","5","7"])
                    if len(roi_AU) > 0:
                        roi_choice = random.choice(list(roi_AU.keys()))
                        AU_choice = random.choice(roi_AU[roi_choice])
                        already_use_roi.add(roi_choice)
                    if AU_choice not in print_tuple:
                        print_tuple.append(AU_choice)
                final_tuple_list.add(tuple(sorted(print_tuple, key=lambda e:int(e))))
        for t in final_tuple_list:
            print(t)

