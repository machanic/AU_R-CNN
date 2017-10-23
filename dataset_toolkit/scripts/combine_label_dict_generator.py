import os
import sys
sys.path.insert(0, '/home/machen/face_expr')
import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU
from collections import defaultdict
from itertools import combinations


# 结论：同一个区域会有多个AU同时发生的存在, AU>=40 后都为9，未标注
def check_BP4D_multi_label(label_file_dir, au_mask_dir):
    au_couple_dict = get_zip_ROI_AU()
    combine_AU = defaultdict(set)
    all_AU = set()
    all_co_occur_AU = defaultdict(int)
    for file_name in os.listdir(label_file_dir):
        subject_name = file_name[:file_name.index("_")]
        sequence_name = file_name[file_name.index("_")+1:file_name.rindex(".")]
        AU_column_idx = {}
        co_occur_AU = {}
        with open(label_file_dir+"/" +file_name, "r") as au_file_obj:
            for idx, line in enumerate(au_file_obj):
                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue

                lines = line.split(",")
                frame = lines[0]
                non_label_lst = [int(lines[AU_column_idx[AU]]) for AU in config.AU_ROI.keys() if int(AU) >= 40 and int(lines[AU_column_idx[AU]]) != 9]
                if len(non_label_lst) > 0:
                    print(" big AU is not 9! in {}".format(label_file_dir+"/" +file_name))
                    print("{}".format(non_label_lst))
                au_label_dict = {AU: int(lines[AU_column_idx[AU]]) for AU in config.AU_ROI.keys() \
                                 if int(lines[AU_column_idx[AU]]) == 1} # 注意只生成=1的字典，而不要=9的字典，就是只要非unknown的AU

                for AU_tuple in combinations(au_label_dict.keys(), 2):
                    co_occur_AU[tuple(sorted(map(int,AU_tuple)))] = 1

                all_AU.update(au_label_dict.keys())
                au_mask_dict = {AU: "{0}/{1}/{2}/{3}_AU_{4}.png".format(au_mask_dir,
                                                                        subject_name, sequence_name,
                                                                        frame, ",".join(au_couple_dict[AU])) \
                                for AU in au_label_dict.keys()}

                for AU, mask_path in au_mask_dict.items():
                    combine_AU[mask_path].add(AU)
        for AU_tuple, count in co_occur_AU.items():
            all_co_occur_AU[AU_tuple] += count
    AU_counter = defaultdict(int)
    for mask_path, combine_AU in combine_AU.items():
        AU_counter[tuple(sorted(combine_AU))] += 1
    # for AU, count in sorted(AU_counter.items(), key=lambda e: e[1],reverse=True):
    #     print("{0} {1}".format(AU, count))
    return AU_counter, all_AU, all_co_occur_AU

if __name__ == "__main__":
    combine_AU_counter, all_AU, co_occur_AU = check_BP4D_multi_label(config.DATA_PATH["BP4D"]+"/AUCoding/", config.AU_REGION_MASK_PATH["BP4D"])
    start_combine_no = 100
    with open("/home/machen/dataset/BP4D/label_dict.txt", "w") as file_obj:
        for AU in sorted(map(int, all_AU)):
            file_obj.write("{0} {1}\n".format(AU, AU))
        for AU_combine in combine_AU_counter.keys():
            if len(AU_combine) < 2: continue
            file_obj.write("{0} {1}\n".format(start_combine_no, ",".join(AU_combine)))
            start_combine_no+=1
        file_obj.flush()
    import config
    JPML_RELATION = config.AU_RELATION_JPML
    for AU_tuple, count in sorted(co_occur_AU.items(), key=lambda e:e[1], reverse=True):
        if AU_tuple not in JPML_RELATION["positive"] and count > 50:
            JPML_RELATION["positive"].append(AU_tuple)
    negative_relation = []
    for AU_tuple in JPML_RELATION["negative"]:
        print(AU_tuple)
        if AU_tuple not in JPML_RELATION["positive"]:
            negative_relation.append(AU_tuple)
    JPML_RELATION["negative"] = negative_relation
    print(JPML_RELATION)