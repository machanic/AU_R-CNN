import os
import config
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict
import math
# Leave one subject out
# data_format:
# image_path    label1,label2,?label_ignore1,?label_ignore2  #/orig_img_path  BP4D/DISFA
# tab split, #/orig_img_path means '#': orignal img_path not transform, orig_img_path is from transform,so first image_path is transformed image

# 这个函数暂时没用到
def get_BP4D_AU_intensity():
    video_frame_AU = defaultdict(dict)
    root_dir = config.DATA_PATH["BP4D"] + "AU-Intensity-Codes3.0"
    for AU in os.listdir(root_dir):
        for csv_file_name in os.listdir(root_dir + os.sep + AU):
            AU = AU[2:]
            video_name = csv_file_name[:csv_file_name.rindex("_")]
            with open(root_dir + os.sep + AU + os.sep + csv_file_name, "r") as file_obj:
                for line in file_obj:
                    frame, AU_intensity = line.strip().split(",")
                    frame = int(frame)
                    AU_intensity = int(AU_intensity)
                    if frame not in video_frame_AU[video_name]:
                        video_frame_AU[video_name][frame] = dict()
                    video_frame_AU[video_name][frame][AU] = AU_intensity
    return video_frame_AU

def single_AU_RCNN_BP4D_subject_id_file(idx_folder_path, kfold=None,  validation_size=3000): # partition_path is dict{"trn":..., "valid":xxx}

    for BP4D_AU in config.paper_use_BP4D:
        full_pretrain = set()
        subject_video = defaultdict(dict)  # key is subject id
        for file_name in os.listdir(config.DATA_PATH["BP4D"] + "/AUCoding"):
            if not file_name.endswith(".csv"): continue
            subject_name = file_name.split("_")[0]
            sequence_name = file_name[file_name.rindex("_") + 1:file_name.rindex(".")]
            video_dir = config.TRAINING_PATH["BP4D"] +os.sep + subject_name + os.sep + sequence_name
            first_frame_file_name = os.listdir(video_dir)[0]
            first_frame_file_name = first_frame_file_name[:first_frame_file_name.rindex(".")]
            frame_len = len(first_frame_file_name)
            AU_column_idx = dict()
            print("reading:{}".format("{0}/{1}".format(config.DATA_PATH["BP4D"] + "/AUCoding", file_name)))

            with open("{0}/{1}".format(config.DATA_PATH["BP4D"] + "/AUCoding", file_name), "r") as au_file_obj:
                for idx, line in enumerate(au_file_obj):
                    if idx == 0:  # header specify Action Unit
                        for col_idx, _AU in enumerate(line.split(",")[1:]):
                            AU_column_idx[_AU] = col_idx + 1  # read header
                        continue  # read head over , continue

                    lines = line.split(",")
                    frame = lines[0].zfill(frame_len)
                    img_file_path = video_dir + os.sep + frame+".jpg"
                    if os.path.exists(img_file_path):
                        AU_set = set()
                        if int(lines[AU_column_idx[BP4D_AU]]) == 1:
                            AU_set.add(BP4D_AU)
                        if len(AU_set) == 0:
                            AU_set.add("0")  # 该frame没有AU
                        if sequence_name not in subject_video[subject_name]:
                            subject_video[subject_name][sequence_name] = list()
                        subject_video[subject_name][sequence_name].append({"img_path":img_file_path, "AU_label":AU_set,
                                                                           "database":"BP4D",
                                                                           "frame":frame, "video_name": "{0}_{1}".format(subject_name, sequence_name)})

        print("reading AU-coding file done")
        subject_name_ls = np.array(list(subject_video.keys()), dtype=str)
        if kfold is not None:
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 0
            folder_path = "{0}/AU_{1}/{2}_fold".format(idx_folder_path, BP4D_AU, kfold)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for train_index, test_index in kf.split(subject_name_ls):
                i += 1
                train_name_array = subject_name_ls[train_index]
                test_name_array = subject_name_ls[test_index]
                balance_line = defaultdict(list)  # AU : write_lines

                for subject_name in train_name_array:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_set = video_info["AU_label"]
                            img_file_path = os.sep.join(video_info["img_path"].split(os.sep)[-3:])
                            AU_set_str = ",".join(AU_set)
                            line = "{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,  orig_from_path, "BP4D")
                            full_pretrain.add(line)
                            balance_line[AU_set_str].append(line)

                ratio = max(len(lines) for lines in balance_line.values()) // min(
                    len(lines) for lines in balance_line.values())
                new_lines = []
                for AU_set_str, lines in balance_line.items():
                    if AU_set_str != "0":
                        for _ in range(ratio):
                            for line in lines:
                                new_lines.append(line)
                balance_line[BP4D_AU].extend(new_lines)
                with open("{0}/id_trainval_{1}.txt".format(folder_path, i), "w") as file_obj:
                    for AU_set_str, lines in balance_line.items():
                        for line in lines:
                            file_obj.write(line)
                    file_obj.flush()
                validate_lines = []
                with open("{0}/id_test_{1}.txt".format(folder_path, i), "w") as file_obj:
                    for subject_name in test_name_array:
                        for info_dict in subject_video[subject_name].values():
                            for video_info in info_dict:
                                orig_from_path = "#"
                                AU_set = video_info["AU_label"]
                                img_file_path = os.sep.join(video_info["img_path"].split(os.sep)[-3:])
                                AU_set_str = ",".join(AU_set)
                                line = "{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,  orig_from_path, "BP4D")
                                validate_lines.append(line)
                                full_pretrain.add(line)
                                file_obj.write("{}".format(line))
                    file_obj.flush()

                validate_lines = np.random.choice(validate_lines, validation_size, replace=False)
                with open("{0}/id_valid_{1}.txt".format(folder_path, i), "w") as file_obj:
                    for line in validate_lines:
                        file_obj.write("{}".format(line))
                    file_obj.flush()
        with open("{0}/full_pretrain.txt".format(folder_path), "w") as file_obj:
            for line in full_pretrain:
                file_obj.write(line)
            file_obj.flush()

def single_AU_DISFA_subject_id_file(idx_folder_path, kfold=None, partition_file_path=None):
    for DISFA_AU in config.paper_use_DISFA:
        DISFA_base_dir = config.DATA_PATH["DISFA"]
        label_file_dir = DISFA_base_dir + "/ActionUnit_Labels/"
        subject_video = defaultdict(dict)  # key is subject id
        orientations = ["Left", "Right"]

        for video_name in os.listdir(label_file_dir):
            frame_label = {}
            for label_file_name in os.listdir(label_file_dir+os.sep+video_name):
                AU = label_file_name[label_file_name.index("au") + 2: label_file_name.rindex(".")]
                if AU != DISFA_AU:
                    continue
                with open(label_file_dir+os.sep+video_name+os.sep+label_file_name, "r") as file_obj:
                    for line in file_obj:
                        line = line.strip()
                        if line:
                            frame, AU_intensity = line.split(",")
                            AU_intensity = int(AU_intensity)
                            if frame not in frame_label:
                                frame_label[frame] = set()
                            if AU_intensity >= 3:   # FIXME 是否需要改为>= 3?
                                frame_label[frame].add(AU)
            for orientation in orientations:
                img_folder = DISFA_base_dir + "/Img_{}Camera".format(orientation)
                for frame, AU_set in sorted(frame_label.items(), key=lambda e:int(e[0])):
                    if orientation not in subject_video[video_name]:
                        subject_video[video_name][orientation] = []
                    img_file_path = img_folder + "/" + video_name + "/" + frame + ".jpg"
                    if os.path.exists(img_file_path):
                        subject_video[video_name][orientation].append({"img_path":img_file_path, "AU_label":AU_set,
                                                                       "database":"DISFA"})

        subject_name_ls = np.array(list(subject_video.keys()), dtype=str)
        if kfold is not None:
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 0
            for train_index, test_index in kf.split(subject_name_ls):
                i += 1
                train_name_array = subject_name_ls[train_index]
                test_name_array = subject_name_ls[test_index]
                folder_path = "{0}/AU_{1}/{2}_fold".format(idx_folder_path, DISFA_AU, kfold)
                print(folder_path)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                balance_line = defaultdict(list)  # DISFA_AU : write_lines

                for video_name in train_name_array:
                    for orientation, video_info_lst in subject_video[video_name].items():
                        for video_info in video_info_lst:
                            img_file_path = video_info["img_path"]
                            img_file_path = os.sep.join(img_file_path.split("/")[-3:])
                            AU_set_str = ",".join(video_info["AU_label"])
                            if len(video_info["AU_label"]) == 0:
                                AU_set_str = "0"
                            orig_from_path = "#"
                            line = "{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,orig_from_path,video_info["database"])
                            balance_line[AU_set_str].append(line)
                ratio = max(len(lines) for lines in balance_line.values()) // min(len(lines) for lines in balance_line.values())

                new_lines = []
                for AU_set_str, lines in balance_line.items():
                    if AU_set_str != "0":
                        for _ in range(ratio):
                            for line in lines:
                                new_lines.append(line)
                balance_line[DISFA_AU].extend(new_lines)
                with open("{0}/id_trainval_{1}.txt".format(folder_path, i), "w") as file_obj:
                    for AU_set_str, lines in balance_line.items():
                        for line in lines:
                            file_obj.write(line)
                    file_obj.flush()
                with open("{0}/id_test_{1}.txt".format(folder_path, i), "w") as file_obj:
                    for video_name in test_name_array:
                        for orientation, video_info_lst in subject_video[video_name].items():
                            for video_info in video_info_lst:
                                img_file_path = video_info["img_path"]
                                img_file_path = os.sep.join(img_file_path.split("/")[-3:])
                                AU_set_str = ",".join(video_info["AU_label"])
                                if len(video_info["AU_label"]) == 0:
                                    AU_set_str = "0"
                                orig_from_path = "#"
                                file_obj.write("{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,orig_from_path,video_info["database"]))
                    file_obj.flush()
                with open("{0}/id_valid_{1}.txt".format(folder_path, i), "w") as file_obj:
                    for video_name in test_name_array:
                        for orientation, video_info_lst in subject_video[video_name].items():
                            for video_info in video_info_lst:
                                img_file_path = video_info["img_path"]
                                img_file_path = os.sep.join(img_file_path.split("/")[-3:])
                                AU_set_str = ",".join(video_info["AU_label"])
                                if len(video_info["AU_label"]) == 0:
                                    AU_set_str = "0"
                                orig_from_path = "#"
                                file_obj.write("{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,orig_from_path,video_info["database"]))
                    file_obj.flush()


def gen_BP4D_subject_id_file(idx_folder_path, kfold=None, partition_path=None, validation_size=3000): # partition_path is dict{"trn":..., "valid":xxx}
    subject_video = defaultdict(dict)  # key is subject id
    BP4D_lines = set()
    pretrained_full = set()
    for file_name in os.listdir(config.DATA_PATH["BP4D"] + "/AUCoding"):
        if not file_name.endswith(".csv"): continue
        subject_name = file_name.split("_")[0]
        sequence_name = file_name[file_name.rindex("_") + 1:file_name.rindex(".")]
        video_dir = config.TRAINING_PATH["BP4D"] +os.sep + subject_name + os.sep + sequence_name
        first_frame_file_name = os.listdir(video_dir)[0]
        first_frame_file_name = first_frame_file_name[:first_frame_file_name.rindex(".")]
        frame_len = len(first_frame_file_name)
        AU_column_idx = dict()
        print("reading:{}".format("{0}/{1}".format(config.DATA_PATH["BP4D"] + "/AUCoding", file_name)))

        with open("{0}/{1}".format(config.DATA_PATH["BP4D"] + "/AUCoding", file_name), "r") as au_file_obj:
            for idx, line in enumerate(au_file_obj):
                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue

                lines = line.split(",")
                frame = lines[0].zfill(frame_len)
                img_file_path = video_dir + os.sep + frame+".jpg"
                if os.path.exists(img_file_path):
                    AU_set = set()
                    for AU in config.AU_ROI.keys():
                        if int(lines[AU_column_idx[AU]]) == 1:
                            AU_set.add(AU)
                        elif int(lines[AU_column_idx[AU]]) == 9:
                            AU_set.add("?{}".format(AU))

                    if len(AU_set) == 0 or not list(filter(lambda e: not e.startswith("?"), AU_set)):
                        AU_set.add("0")  # 该frame没有AU
                    if sequence_name not in subject_video[subject_name]:
                        subject_video[subject_name][sequence_name] = list()
                    subject_video[subject_name][sequence_name].append({"img_path":img_file_path, "AU_label":AU_set, "database":"BP4D"})

    print("reading AU-coding file done")
    subject_name_ls = np.array(list(subject_video.keys()), dtype=str)
    if kfold is not None:
        kf = KFold(n_splits=kfold, shuffle=True)
        i = 0
        folder_path = "{0}/{1}_fold".format(idx_folder_path, kfold)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for train_index, test_index in kf.split(subject_name_ls):
            i += 1
            train_name_array = subject_name_ls[train_index]
            test_name_array = subject_name_ls[test_index]

            with open("{0}/id_trainval_{1}.txt".format(folder_path, i), "w") as file_obj:
                for subject_name in train_name_array:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_set = video_info["AU_label"]
                            img_file_path = os.sep.join(video_info["img_path"].split(os.sep)[-3:])
                            AU_set_str = ",".join(AU_set)
                            line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_set_str,  orig_from_path, "BP4D")
                            pretrained_full.add(line)
                            file_obj.write("{}\n".format(line))
                file_obj.flush()

            validate_lines = []
            with open("{0}/id_test_{1}.txt".format(folder_path, i), "w") as file_obj:
                for subject_name in test_name_array:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_set = video_info["AU_label"]
                            img_file_path = os.sep.join(video_info["img_path"].split(os.sep)[-3:])
                            AU_set_str = ",".join(AU_set)
                            line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_set_str,  orig_from_path, "BP4D")
                            validate_lines.append(line)
                            pretrained_full.add(line)
                            file_obj.write("{}\n".format(line))
                file_obj.flush()

            validate_lines = np.random.choice(validate_lines, validation_size, replace=False)
            with open("{0}/id_valid_{1}.txt".format(folder_path, i), "w") as file_obj:
                for line in validate_lines:
                    file_obj.write("{}\n".format(line))
                file_obj.flush()
        with open("{}/full_pretrain.txt".format(folder_path), "w") as file_obj:
            for line in pretrained_full:
                file_obj.write("{}\n".format(line))
            file_obj.flush()



    if partition_path is not None:
        trn_subject_name = []
        validate_subject_name = []
        trn_subject_file_path = partition_path["trn"]
        valid_subject_file_path = partition_path["valid"]
        with open(trn_subject_file_path, "r") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    trn_subject_name.append(line)
        with open(valid_subject_file_path, "r") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    validate_subject_name.append(line)
        folder_path = idx_folder_path + "/official_partition"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open("{}/id_train.txt".format(folder_path), "w") as file_obj:
            for subject_name in trn_subject_name:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_set = video_info["AU_label"]
                            img_file_path = os.sep.join(video_info["img_path"].split(os.sep)[-3:])
                            AU_set_str = ",".join(AU_set)
                            line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_set_str,  orig_from_path, video_info["database"])
                            file_obj.write("{}\n".format(line))
            file_obj.flush()
        with open("{}/id_validate.txt".format(folder_path), "w") as file_obj:
            for subject_name in validate_subject_name:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_set = video_info["AU_label"]
                            img_file_path = os.sep.join(video_info["img_path"].split(os.sep)[-3:])
                            AU_set_str = ",".join(AU_set)
                            line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_set_str,  orig_from_path, video_info["database"])
                            file_obj.write("{}\n".format(line))
            file_obj.flush()
    return BP4D_lines


def gen_DISFA_subject_id_file(idx_folder_path, kfold=None, partition_file_path=None):
    DISFA_base_dir = config.DATA_PATH["DISFA"]
    label_file_dir = DISFA_base_dir + "/ActionUnit_Labels/"
    subject_video = defaultdict(dict)  # key is subject id
    orientations = ["Left", "Right"]

    for video_name in os.listdir(label_file_dir):
        frame_label = {}
        for label_file_name in os.listdir(label_file_dir+os.sep+video_name):
            AU = label_file_name[label_file_name.index("au") + 2: label_file_name.rindex(".")]
            with open(label_file_dir+os.sep+video_name+os.sep+label_file_name, "r") as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if line:
                        frame, AU_intensity = line.split(",")
                        AU_intensity = int(AU_intensity)
                        if frame not in frame_label:
                            frame_label[frame] = set()
                        if AU_intensity >= 1:  # FIXME 是否需要改为>= 3?
                            frame_label[frame].add(AU)
        for orientation in orientations:
            img_folder = DISFA_base_dir + "/Img_{}Camera".format(orientation)
            for frame, AU_set in sorted(frame_label.items(), key=lambda e:int(e[0])):
                if orientation not in subject_video[video_name]:
                    subject_video[video_name][orientation] = []
                img_file_path = img_folder + "/" + video_name + "/" + frame + ".jpg"
                if os.path.exists(img_file_path):
                    subject_video[video_name][orientation].append({"img_path":img_file_path, "AU_label":AU_set,
                                                                   "database":"DISFA"})

    subject_name_ls = np.array(list(subject_video.keys()), dtype=str)
    if kfold is not None:
        kf = KFold(n_splits=kfold, shuffle=True)
        i = 0
        for train_index, test_index in kf.split(subject_name_ls):
            i += 1
            train_name_array = subject_name_ls[train_index]
            test_name_array = subject_name_ls[test_index]
            folder_path = "{0}/{1}_fold".format(idx_folder_path, kfold)
            print(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open("{0}/id_trainval_{1}.txt".format(folder_path, i), "w") as file_obj:

                for video_name in train_name_array:
                    for orientation, video_info_lst in subject_video[video_name].items():
                        for video_info in video_info_lst:
                            img_file_path = video_info["img_path"]
                            img_file_path = os.sep.join(img_file_path.split("/")[-3:])
                            AU_set_str = ",".join(video_info["AU_label"])
                            if len(video_info["AU_label"]) == 0:
                                AU_set_str = "0"
                            orig_from_path = "#"
                            file_obj.write("{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,orig_from_path,video_info["database"]))
                file_obj.flush()
            with open("{0}/id_test_{1}.txt".format(folder_path, i), "w") as file_obj:
                for video_name in test_name_array:
                    for orientation, video_info_lst in subject_video[video_name].items():
                        for video_info in video_info_lst:
                            img_file_path = video_info["img_path"]
                            img_file_path = os.sep.join(img_file_path.split("/")[-3:])
                            AU_set_str = ",".join(video_info["AU_label"])
                            if len(video_info["AU_label"]) == 0:
                                AU_set_str = "0"
                            orig_from_path = "#"
                            file_obj.write("{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,orig_from_path,video_info["database"]))
                file_obj.flush()
            with open("{0}/id_valid_{1}.txt".format(folder_path, i), "w") as file_obj:
                for video_name in test_name_array:
                    for orientation, video_info_lst in subject_video[video_name].items():
                        for video_info in video_info_lst:
                            img_file_path = video_info["img_path"]
                            img_file_path = os.sep.join(img_file_path.split("/")[-3:])
                            AU_set_str = ",".join(video_info["AU_label"])
                            if len(video_info["AU_label"]) == 0:
                                AU_set_str = "0"
                            orig_from_path = "#"
                            file_obj.write("{0}\t{1}\t{2}\t{3}\n".format(img_file_path, AU_set_str,orig_from_path,video_info["database"]))
                file_obj.flush()



if __name__ == "__main__":
    from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
    #
    # adaptive_AU_database("BP4D")
    # partition = {"trn":"/home/machen/dataset/BP4D/idx/trn_partition.txt",
    #              "valid":"/home/machen/dataset/BP4D/idx/validate_partition.txt"}
    # gen_BP4D_subject_id_file("{0}/{1}".format(config.DATA_PATH["BP4D"], "idx"), kfold=10, validation_size=1000)
    adaptive_AU_database("DISFA")
    # single_AU_RCNN_BP4D_subject_id_file("{0}/{1}".format(config.ROOT_PATH + os.sep+"/BP4D/", "idx"), kfold=3)
    gen_DISFA_subject_id_file("{0}/{1}".format(config.ROOT_PATH + os.sep+"/DISFA_1/", "idx"), kfold=3)
    # gen_BP4D_subject_id_file("{0}/{1}".format(config.DATA_PATH["BP4D"], "idx"), kfold=10)
    # gen_BP4D_subject_id_file("{0}/{1}".format(config.DATA_PATH["BP4D"], "idx"), kfold=3)
    # print("done")