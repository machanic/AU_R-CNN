import sys
from collections import defaultdict
import bisect
from optparse import OptionParser
import os
import itertools


def read_BP4D_AU_file(folder):
    line_ls = []
    file_path = folder + os.sep + "full_pretrain.txt"

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
    options, args = opt.parse_args()
    data_path = options.data_path
    if data_path is None:
        opt.error("error, data_path not given!")
    min_support = options.min_support
    if min_support is None:
        opt.error("error, min_support not given!")
    output = options.output
    if output != "stdout":
        sys.stdout = open(output, 'w')

    data = read_BP4D_AU_file(data_path)
    import pyfpgrowth
    from collections import defaultdict
    import config
    from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
    au_couple_dict = get_zip_ROI_AU()
    AU_idx  = {"0":999}
    for idx, couple in enumerate(list(set(au_couple_dict.values()))):
        for AU in couple:
            AU_idx[AU] = idx
    freq_seqs = pyfpgrowth.find_frequent_patterns(data, 1000)

    # output result frequent items
    all_print_tuple = set()
    for tu in sorted(freq_seqs.items(), key=lambda e:e[1], reverse=True):
        idx_set = set(AU_idx[entry] for entry in tu[0])
        if len(idx_set) == 1:
            continue
        print_dict = defaultdict(list)
        for entry in tu[0]:
            print_dict[AU_idx[entry]].append(entry)
        for entry in itertools.product(*print_dict.values()):
            print_str = ",".join(map(str, sorted(map(int, list(entry)))))
            all_print_tuple.add(print_str)
    file_obj = open(os.path.dirname(os.path.dirname(data_path)) + os.sep + "frequent_pattern_AU.txt",
                    "w")  # save to idx folder
    for print_str in all_print_tuple:
        file_obj.write("{}\n".format(print_str))
    file_obj.flush()
    file_obj.close()
        # pass
