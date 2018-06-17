import os

from collections import defaultdict
import config

def gen_dict(root_dir):
    all_count = 0
    file_ls = ["id_trainval_1.txt", "id_test_1.txt"]
    AU_count = defaultdict(int)
    for file_name in file_ls:
        with open(root_dir + os.sep + file_name, "r") as file_obj:
            for line in file_obj:
                all_count += 1
                lines = line.strip().split()
                AU_str = lines[1]
                for AU in AU_str.split(","):
                    if AU != "0":
                        AU_count[AU] += 1

    for AU,count in sorted(AU_count.items(),key=lambda e:int(e[0]),reverse=False):
        print("{0}={1} {2}".format(AU,count, (all_count - count)/count ))

if __name__ == "__main__":
    gen_dict(config.DATA_PATH["DISFA"]+"/idx/3_fold/")