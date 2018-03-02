from collections import defaultdict

def read_my_index_file(datapath_ls):
    subjectseqname_line_dict = defaultdict(list)
    for datapath in datapath_ls:
        with open(datapath, "r") as file_obj:
            for line in file_obj:
                lines = line.split()  # eg. F023/T4/0935.jpg	14,12,6,10,24,2,15,23,7	#	BP4D
                old_key = lines[0].split("/")
                new_key = "{0}".format(old_key[0])
                subjectseqname_line_dict[new_key].append(line.rstrip())
    return subjectseqname_line_dict

def generate_file(liwei_source_path, target_path, subjectseqname_line_dict, need_write=True):
    if need_write:
        target_file_obj = open(target_path, "w")
        already_use_subject = set()
        with open(liwei_source_path, "r") as file_obj:
            for line in file_obj:
                key = line.split("->")[0].split("_")[0]
                if key in already_use_subject:
                    continue
                already_use_subject.add(key)
                new_lines = subjectseqname_line_dict[key]
                for new_line in new_lines:
                    target_file_obj.write("{}\n".format(new_line))
        target_file_obj.flush()
        target_file_obj.close()
    else:
        already_use_subject = set()
        with open(liwei_source_path, "r") as file_obj:
            for line in file_obj:
                key = line.split("->")[0].split("_")[0]
                already_use_subject.add(key)
    return already_use_subject

def judge_rest_fold(subjectseqname_line_dict, fold_2_subject_name_set, fold_3_subject_name_set, target_path):
    all_set = set(subjectseqname_line_dict.keys())
    rest_subject_name_set = all_set - fold_2_subject_name_set
    rest_subject_name_set = rest_subject_name_set - fold_3_subject_name_set
    target_file_obj = open(target_path, "w")
    for subject in rest_subject_name_set:
        new_lines = subjectseqname_line_dict[subject]
        for new_line in new_lines:
            target_file_obj.write("{}\n".format(new_line))
    target_file_obj.flush()
    target_file_obj.close()

def judge_rest_trainfold(test_path, subjectseqname_line_dict, target_path):
    test_subjectname = set()
    with open(test_path, "r") as file_obj:
        for line in file_obj:
            key = line.split()[0].split("/")[0]
            test_subjectname.add(key)

    all_set = set(subjectseqname_line_dict.keys())

    rest_subject_name_set = all_set - test_subjectname
    target_file_obj = open(target_path, "w")
    for subject in rest_subject_name_set:
        new_lines = subjectseqname_line_dict[subject]
        for new_line in new_lines:
            target_file_obj.write("{}\n".format(new_line))
    target_file_obj.flush()
    target_file_obj.close()

def check_liwei_and_me(liwei_file, me_file):
    me_set = set()
    with open(me_file, "r") as file_obj:
        for line in file_obj:
            if not line:continue
            key = line.split()[0]
            me_set.add(key)
    liwei_set = set()
    with open(liwei_file, "r") as file_obj:
        for line in file_obj:
            if not line: continue
            key = "/".join(line.split("->")[0].split("_"))
            liwei_set.add(key)
    me_not_have = list(set(liwei_set).difference(set(me_set)))
    print("me not have!!!!!!!")
    for me_not in me_not_have:
        print(me_not)
    print("liwei not have!!!!!")
    liwei_not_have = list(set(me_set).difference(set(liwei_set)))
    for liwei_not in liwei_not_have:
        print(liwei_not)
    print("-----------------")

if __name__ == "__main__":
    my_data_file_list = ["G:/Facial AU detection dataset/BP4D/idx/3_fold/id_trainval_1.txt",
                         "G:/Facial AU detection dataset/BP4D/idx/3_fold/id_test_1.txt"]
    subjectseqname_line_dict = read_my_index_file(my_data_file_list)
    test_path = "G:/Facial AU detection dataset/BP4D/idx/liwei_3_fold/id_test_1.txt"
    target_path = "G:/Facial AU detection dataset/BP4D/idx/liwei_3_fold/id_trainval_1.txt"
    fold_2_subject_name_set = generate_file("G:/Facial AU detection dataset/BP4D/idx/BP4D_ts_fold2.txt", target_path, subjectseqname_line_dict,
                  need_write=False)
    fold_3_subject_name_set = generate_file("G:/Facial AU detection dataset/BP4D/idx/BP4D_ts_fold3.txt", target_path, subjectseqname_line_dict,
                  need_write=False)

    check_liwei_and_me("G:/Facial AU detection dataset/BP4D/idx/BP4D_ts_fold3.txt", "G:/Facial AU detection dataset/BP4D/idx/liwei_3_fold/id_test_3.txt")

