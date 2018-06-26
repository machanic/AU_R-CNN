import os
import config
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict

# Leave one subject out
# data_format:
# image_path    label1,label2,?label_ignore1,?label_ignore2  #/orig_img_path  BP4D/DISFA
# tab split, #/orig_img_path means '#': orignal img_path not transform, orig_img_path is from transform,so first image_path is transformed image

def get_BP4D_AU_intensity():
    video_frame_AU = defaultdict(dict)
    root_dir = config.DATA_PATH["BP4D"] + "AU-Intensity-Codes3.0"
    for AU in os.listdir(root_dir):
        for csv_file_name in os.listdir(root_dir + os.path.sep + AU):
            AU_ = AU[2:]
            video_name = csv_file_name[:csv_file_name.rindex("_")]
            with open(root_dir + os.path.sep + AU + os.path.sep + csv_file_name, "r") as file_obj:
                for line in file_obj:
                    frame, AU_intensity = line.strip().split(",")
                    frame = int(frame)
                    AU_intensity = int(AU_intensity)
                    if frame not in video_frame_AU[video_name]:
                        video_frame_AU[video_name][frame] = dict()
                    video_frame_AU[video_name][frame][AU_] = AU_intensity
    return video_frame_AU


def gen_BP4D_subject_id_file(idx_folder_path, video_frame_AU,
                             kfold=None):
    subject_video = defaultdict(dict)  # key is subject id
    for file_name in os.listdir(config.DATA_PATH["BP4D"] + "/AUCoding"):
        if not file_name.endswith(".csv"):
            continue
        subject_name = file_name.split("_")[0]
        sequence_name = file_name[file_name.rindex("_") + 1:  file_name.rindex(".")]
        video_dir = config.RGB_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name
        first_frame_file_name = os.listdir(video_dir)[0]
        first_frame_file_name = first_frame_file_name[:first_frame_file_name.rindex(".")]
        frame_len = len(first_frame_file_name)
        AU_column_idx = dict()
        # print("reading:{}".format("{0}/{1}".format(config.DATA_PATH["BP4D"] + "/AUCoding", file_name)))

        with open("{0}/{1}".format(config.DATA_PATH["BP4D"] + "/AUCoding", file_name), "r") as au_file_obj:
            for idx, line in enumerate(au_file_obj):
                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue

                lines = line.split(",")
                frame = lines[0].zfill(frame_len)
                video_name = subject_name + "_" + sequence_name
                try:
                    AU_intensity = video_frame_AU[video_name][int(frame)]
                except KeyError:
                    print("video : {0} 's frame {1} cause error".format(video_name,frame))
                    continue
                AU_intensity_vector = np.zeros(5, dtype=np.int32)
                for idx, (AU, intensity) in enumerate(sorted(AU_intensity.items(), key=lambda e: int(e[0]))):
                    AU_intensity_vector[idx] = intensity
                img_file_path = video_dir + os.sep + frame + ".jpg"
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
                    subject_video[subject_name][sequence_name].append({"img_path": img_file_path,
                                                                       "AU_label": AU_set,
                                                                       "AU_intensity": ",".join(map(str, AU_intensity_vector)),
                                                                       "database": "BP4D"})

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

            with open("{0}/intensity_trainval_{1}.txt".format(folder_path, i), "w") as file_obj:
                for subject_name in train_name_array:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_intensity = video_info["AU_intensity"]
                            img_file_path = os.path.sep.join(video_info["img_path"].split(os.path.sep)[-3:])
                            line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_intensity,  orig_from_path, "BP4D")
                            file_obj.write("{}\n".format(line))
                file_obj.flush()

            with open("{0}/intensity_test_{1}.txt".format(folder_path, i), "w") as file_obj:
                for subject_name in test_name_array:
                    for info_dict in subject_video[subject_name].values():
                        for video_info in info_dict:
                            orig_from_path = "#"
                            AU_intensity = video_info["AU_intensity"]
                            img_file_path = os.path.sep.join(video_info["img_path"].split(os.path.sep)[-3:])
                            line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_intensity,  orig_from_path, "BP4D")
                            file_obj.write("{}\n".format(line))
                file_obj.flush()



if __name__ == "__main__":
    from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

    adaptive_AU_database("BP4D")
    video_frame_AU = get_BP4D_AU_intensity()  # video_name, frame, AU => intensity
    gen_BP4D_subject_id_file("{0}/{1}".format(config.DATA_PATH["BP4D"], "idx"), video_frame_AU, 3, )
    # single_AU_RCNN_BP4D_subject_id_file("{0}/{1}".format(config.ROOT_PATH + os.sep+"/BP4D/", "idx"), kfold=3)
