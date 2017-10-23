from  dataset_toolkit.data_reader import DataFactory
import os
import config
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict
from dataset_toolkit.scripts.label_balance import database_enhance_balance, AU_calcuate_from_dataset



def gen_video_fold_id_file(database, folder_path, kfold):


    BP4D_data_reader = DataFactory.get_data_reader(database)
    BP4D_data_reader.read()
    kfold_samples = DataFactory.kfold_video_split(database, kfold, True)
    for i in range(kfold):
        trn_data_reader = kfold_samples[i]["trn"]
        test_data_reader = kfold_samples[i]["test"]
        print("train: {0} {1}".format(i, len(trn_data_reader.images)))
        print("test: {0} {1}".format(i, len(test_data_reader.images)))
        with open("{0}/id_trainval_{1}.txt".format(folder_path,i), "w") as file_obj:
            for img_path in trn_data_reader.images:
                subject_name = img_path.split(os.sep)[-3]
                sequence_name = img_path.split(os.sep)[-2]
                img_file = img_path.split(os.sep)[-1]
                frame = img_file[:img_file.rindex(".")]
                id = "{0}/{1}/{2}".format(subject_name, sequence_name, frame)
                file_obj.write(id)
                file_obj.write("\n")
            file_obj.flush()
        with open("{0}/id_test_{1}.txt".format(folder_path, i), "w") as file_obj:
            for img_path in test_data_reader.images:
                subject_name = img_path.split(os.sep)[-3]
                sequence_name = img_path.split(os.sep)[-2]
                img_file = img_path.split(os.sep)[-1]
                frame = img_file[:img_file.rindex(".")]
                id = "{0}/{1}/{2}".format(subject_name, sequence_name, frame)
                file_obj.write(id)
                file_obj.write("\n")
            file_obj.flush()



# Leave one subject out
# data_format:
# image_path    label1,label2,?label_ignore1,?label_ignore2  #/orig_img_path  BP4D/DISFA
# tab split, #/orig_img_path means '#': orignal img_path not transform, orig_img_path is from transform,so first image_path is transformed image
def gen_BP4D_subject_kfold_id_file(database_name, folder_path, kfold, drop_big_label=False):

    BP4D_data_reader = DataFactory.get_data_reader(database_name)


    BP4D_dataset = defaultdict(set)

    for file_name in os.listdir(BP4D_data_reader.AU_occur_dir):
        subject_name = file_name.split("_")[0]
        sequence_name = file_name[file_name.rindex("_") + 1:file_name.rindex(".")]
        video_dir = "BP4D/BP4D_crop/" +os.sep+ subject_name + os.sep+sequence_name
        first_frame_file_name = os.listdir(config.ROOT_PATH+os.sep+video_dir)[0]
        first_frame_file_name = first_frame_file_name[:first_frame_file_name.rindex(".")]
        frame_len = len(first_frame_file_name)
        AU_column_idx = {}
        print("reading:{}".format("{0}/{1}".format(BP4D_data_reader.AU_occur_dir, file_name)))
        with open("{0}/{1}".format(BP4D_data_reader.AU_occur_dir, file_name), "r") as au_file_obj:

            for idx, line in enumerate(au_file_obj):
                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue

                lines = line.split(",")
                frame = lines[0].zfill(frame_len)
                img_file_path = video_dir + os.sep + frame+".jpg"
                if os.path.exists(config.ROOT_PATH + os.sep + img_file_path):
                    AU_set = set()
                    for AU in config.AU_ROI.keys():
                        if int(lines[AU_column_idx[AU]]) == 1:
                            AU_set.add(AU)
                        elif int(lines[AU_column_idx[AU]]) == 9:
                            AU_set.add("?{}".format(AU))

                    if len(AU_set) == 0 or not list(filter(lambda e: not e.startswith("?"), AU_set)):
                        AU_set.add("0")  # 该frame没有AU
                    BP4D_dataset[img_file_path].update(AU_set)

    print("reading AU-coding file done")
    subject_name_ls = set()

    AU_count = AU_calcuate_from_dataset(BP4D_dataset)
    enhance_mix_database, img_from = database_enhance_balance(BP4D_dataset, AU_count, drop_big_label=drop_big_label)

    subject_imgpath_dict = defaultdict(set)
    check_miss_set = set(BP4D_dataset.keys()) - set(enhance_mix_database.keys())
    check_togathor_set = set(BP4D_dataset.keys()) & set(enhance_mix_database.keys())

    print("missing: {0} intersetion:{1} BP4D:{2} enhance_data:{3}".format(len(check_miss_set), len(check_togathor_set), len(set(BP4D_dataset.keys())), len(set(enhance_mix_database.keys()))))
    for enhance_img_path in enhance_mix_database.keys():
        subject_name = enhance_img_path.split(os.sep)[-3]
        subject_name_ls.add(subject_name)
        subject_imgpath_dict[subject_name].add(enhance_img_path)


    subject_name_ls = np.array(list(subject_name_ls))
    kf = KFold(n_splits=kfold, shuffle=True)
    i = 0
    for train_index, test_index in kf.split(subject_name_ls):
        i += 1
        train_name_array = subject_name_ls[train_index]
        test_name_array = subject_name_ls[test_index]
        with open("{0}/id_trainval_{1}.txt".format(folder_path, i), "w") as file_obj:
            for subject_name in train_name_array:
                for img_file_path in subject_imgpath_dict[subject_name]:
                    orig_from_path = "#" if img_file_path not in img_from else img_from[img_file_path]
                    AU_set = enhance_mix_database[img_file_path]
                    AU_set_str = ",".join(AU_set)
                    line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_set_str,  orig_from_path, "BP4D")
                    file_obj.write("{}\n".format(line))
            file_obj.flush()

        with open("{0}/id_test_{1}.txt".format(folder_path, i), "w") as file_obj:
            for subject_name in test_name_array:
                for img_file_path in subject_imgpath_dict[subject_name]:
                    orig_from_path = "#" if img_file_path not in img_from else img_from[img_file_path]
                    video_dir = BP4D_data_reader.img_dir + os.sep + subject_name + os.sep + sequence_name + os.sep
                    AU_set = enhance_mix_database[img_file_path]
                    AU_set_str = ",".join(AU_set)
                    line = "{0}\t{1}\t{2}\t{3}".format(img_file_path, AU_set_str, orig_from_path, "BP4D")
                    file_obj.write("{}\n".format(line))
            file_obj.flush()



if __name__ == "__main__":
    from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

    adaptive_AU_database("BP4D")
    gen_BP4D_subject_kfold_id_file("BP4D", "{0}/{1}".format(config.DATA_PATH["BP4D"], "idx"), kfold=10,drop_big_label=False)
    print("done")


