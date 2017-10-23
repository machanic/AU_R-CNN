import sys
import math
sys.path.insert(0, '/home/machen/face_expr')

from dataset_toolkit.compress_utils import get_zip_ROI_AU
import os
from collections import defaultdict
from config import DATA_PATH,ROOT_PATH
from functools import lru_cache
import copy
import numpy as np
import math
from PIL import Image, ImageEnhance
import config
import multiprocessing as mp
import time

def AU_stats_DISFA(label_file_dir, skip_frame=1):
    AU_count = defaultdict(int)
    img_idx_AU = defaultdict(set)
    for file_name in os.listdir(label_file_dir):
        subject_name = file_name
        for au_file in os.listdir(label_file_dir+"/"+file_name):
            AU = au_file[au_file.index("_au")+3:au_file.rindex(".")]
            with open(label_file_dir + os.sep + file_name + os.sep + au_file, "r") as file_obj:
                for idx, line in enumerate(file_obj):
                    if idx % skip_frame != 0: # 每一行是一个frame图片
                        continue

                    lines = line.strip().split(",")
                    frame = int(lines[0])
                    AU_level = lines[1]
                    if AU_level != "0":
                        AU_count[AU] += 1
                        img_idx_AU["DISFA:{0}/{1}".format(subject_name, frame)].add(AU)

    return AU_count, img_idx_AU

def AU_stats_BP4D(label_file_dir, skip_frame=1):
    '''
    :param label_file_dir: dict which contain all AU label file
    :return: dict = {AU : sample_count}
    '''
    AU_count = defaultdict(int)
    img_idx_AU = defaultdict(set)
    for file_name in os.listdir(label_file_dir):
        subject_name = file_name[:file_name.index("_")]
        sequence_name = file_name[file_name.index("_") + 1:file_name.rindex(".")]
        AU_column_idx = {}
        with open(label_file_dir + "/" + file_name, "r") as au_file_obj:
            for idx, line in enumerate(au_file_obj): # 每行是一帧画面的label

                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue
                if idx % skip_frame != 0:
                    continue
                lines = line.split(",")
                frame = lines[0]
                au_label_set = set([AU for AU in AU_ROI.keys() \
                                 if int(lines[AU_column_idx[AU]]) == 1])

                if len(au_label_set) > 0:
                    img_idx_AU["BP4D:{0}/{1}/{2}".format(subject_name, sequence_name, frame)].update(au_label_set)
                for AU in au_label_set:
                    AU_count[AU] += 1
    return AU_count, img_idx_AU



def AU_repeat_level(level_num, AU_count):
    '''

    :param level_num:  how many AU (repeat) level are there in AU class number
            AU_count: dict key=AU, value=count
    :return: AU_level is a dict, which value is level_index, lower level_index means lower repeat level, higher level means higher repeat level
    '''
    split_list = lambda A, n=level_num: [A[i:i + n] for i in range(0, len(A), n)]
    AU_level = dict()
    # print("sublist:{}".format(split_list(sorted(AU_count.items(), key=lambda e:e[1], reverse=True))))
    for idx, sub in enumerate(split_list(sorted(AU_count.items(), key=lambda e:e[1], reverse=True))):
        for AU, count in sub:
            AU_level[AU] = idx
    return AU_level


# 混合不同数据库，再用锐化之类的增多小类的样本
def database_mix_enhance_balance_check():
    '''
    结论：仍然会被捆绑效应所限，不会平衡
    :return:
    '''
    AU_count, img_idx_AU_DISFA = AU_stats_DISFA("/home/machen/dataset/DISFA/AU_labels/", skip_frame=1)

    AU_count_BP4D, img_idx_AU_BP4D = AU_stats_BP4D("/home/machen/dataset/BP4D/AUCoding/")

    for AU, count in AU_count_BP4D.items():
        AU_count[AU] += count #mix
    level_repeat = {0: 0,
                    1: 4, # 锐化，模糊，原图, 翻转
                    2: 8,} # 左右翻转 x 锐化，模糊，原图 或 左右翻转 x 对比度增强，对比度降低，原图
                    #3: 10} # 左右翻转 x (锐化，模糊，对比度增强，对比度降低, 原图)
    AU_level = AU_repeat_level(math.ceil(len(AU_count)/len(level_repeat)), AU_count)
    img_idx_AU_DISFA.update(img_idx_AU_BP4D) #mix

    mix_database = img_idx_AU_DISFA
    enhance_mix_database = copy.deepcopy(mix_database)
    for img_id, AU_labels in mix_database.items():
        AU_labels = list(AU_labels)
        max_repeat_times = level_repeat[max(AU_level[AU] for AU in AU_labels)]  # 计算每个图的label中需要重复最多的那个AU
        for repeat_idx in range(max_repeat_times):
            enhance_mix_database["{0}_{1}".format(img_id, repeat_idx)].update(AU_labels)
    # now stats enhance_mix_database
    enhance_AU_count = defaultdict(int)
    for img_id, AU_labels in enhance_mix_database.items():
        for AU in AU_labels:
            enhance_AU_count[AU] += 1
    print("remains lost count:{0}  interset count:{1}".format(len(set(list(mix_database.keys())) - set(list(enhance_mix_database.keys()))),
                                                              len(set(list(mix_database.keys())) & set(list(enhance_mix_database.keys())))))
    return enhance_AU_count

def make_dir_not_exists(abs_path):
    dir_name = os.path.dirname(abs_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def AU_calcuate_from_dataset(dataset):
    AU_count = defaultdict(int)
    for img_path, AU_set in dataset.items():
        for AU in AU_set:
            if not AU.startswith("?"):
                AU_count[AU]+=1
    return AU_count


all_new_path_lst = []
enhance_mix_database = defaultdict(set)
img_from = {}
def append_new_path_result(result):
    new_path_lst, orig_AU_labels, img_path = result
    print("a job done:{}".format(img_path))
    all_new_path_lst.extend(new_path_lst)
    for new_img_path in new_path_lst:
        enhance_mix_database[new_img_path].update(orig_AU_labels)  # enhance
        img_from[new_img_path] = img_path


def async_generate_image_save(new_path_lst,
                               repeat_level,
                              subject_name, sequence_name, frame_name,
                              orig_AU_labels, img_path, need_generate):
    transform_func = {
        "sharp": lambda im: ImageEnhance.Sharpness(im).enhance(3.0),
        "fuzzy": lambda im: ImageEnhance.Sharpness(im).enhance(0.0),
        "contrast_improve": lambda im: ImageEnhance.Contrast(im).enhance(1.6),
        "contrast_decrease": lambda im: ImageEnhance.Contrast(im).enhance(0.6),
        "flip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)
    }
    abs_new_path = lambda subject_name, sequence_name, frame, trans: ENHANCE_BALANCE_PATH["BP4D"] + \
                                                                 os.sep + "{0}/{1}/{2}({3}).jpg".format(subject_name,
                                                                                                        sequence_name,
                                                                                                        frame,
                                                                                                        trans)
    relative_new_path = lambda subject_name, sequence_name, frame, trans: "BP4D/BP4D_enhance_balance/{0}/{1}/{2}({3}).jpg".format(subject_name,
                                                                                                        sequence_name,
                                                                                                        frame,
                                                                                                        trans)

    print("opening file :{}".format(img_path))
    im = Image.open(ROOT_PATH+os.sep+img_path)

    if repeat_level == 1:
        enhance_names = ["fuzzy", "sharp", "flip","flip_sharp","flip_fuzzy"]
        for enhance_name in enhance_names:
            rel_im_path = relative_new_path(subject_name, sequence_name, frame_name, enhance_name)
            new_path_lst.append(rel_im_path)
        if need_generate:
            fuzzy_im = transform_func["fuzzy"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "fuzzy")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                fuzzy_im.save(new_im_path)

            sharp_im = transform_func["sharp"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "sharp")
            make_dir_not_exists(new_im_path)

            if not os.path.exists(new_im_path):
                sharp_im.save(new_im_path)

            flip_im = transform_func["flip"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                flip_im.save(new_im_path)

            flip_sharp_im = transform_func["sharp"](flip_im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip_sharp")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                flip_sharp_im.save(new_im_path)

            fuzzy_sharp_im = transform_func["fuzzy"](flip_im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip_fuzzy")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                fuzzy_sharp_im.save(new_im_path)
    elif repeat_level == 2:
        enhance_names = ["fuzzy", "sharp", "contrast+", "contrast-", "flip", "flip_sharp", "flip_fuzzy",
                         "flip_contrast+", "flip_contrast-"]
        for enhance_name in enhance_names:
            rel_im_path = relative_new_path(subject_name, sequence_name, frame_name, enhance_name)
            new_path_lst.append(rel_im_path)
        if need_generate:
            fuzzy_im = transform_func["fuzzy"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "fuzzy")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                fuzzy_im.save(new_im_path)

            sharp_im = transform_func["sharp"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "sharp")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                sharp_im.save(new_im_path)

            contrast_improve_im = transform_func["contrast_improve"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "contrast+")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                contrast_improve_im.save(new_im_path)

            contrast_decrease_im = transform_func["contrast_decrease"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "contrast-")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                contrast_decrease_im.save(new_im_path)

            flip_im = transform_func["flip"](im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                flip_im.save(new_im_path)

            flip_sharp_im = transform_func["sharp"](flip_im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip_sharp")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                flip_sharp_im.save(new_im_path)

            fuzzy_sharp_im = transform_func["fuzzy"](flip_im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip_fuzzy")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                fuzzy_sharp_im.save(new_im_path)

            flip_contrast_improve_im = transform_func["contrast_improve"](flip_im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip_contrast+")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                flip_contrast_improve_im.save(new_im_path)

            flip_contrast_decrease_im = transform_func["contrast_decrease"](flip_im)
            new_im_path = abs_new_path(subject_name, sequence_name, frame_name, "flip_contrast-")
            make_dir_not_exists(new_im_path)
            if not os.path.exists(new_im_path):
                flip_contrast_decrease_im.save(new_im_path)

    return new_path_lst, orig_AU_labels, img_path


# 最终外界程序调用这个函数进行平衡化, 这个函数再平衡化的过程中也会生成锐化模糊过的图片存到硬盘
def database_enhance_balance(dataset, AU_count, drop_big_label=False):
    '''

    :param dataset:  key是img_path, value是AU_label set，包含AU=0，表示背景
    :param AU_count: key是AU str类型, value是count， 包含AU=0，表示背景
    :param drop_big_label:
    :return:
    '''
    level_repeat = {0: 0,
                    1: 5,  # 翻转 x (锐化，模糊，原图) 原图不能算一次多生成的图
                    2: 9, }  # 左右翻转 x (锐化，模糊，对比度增强，对比度降低, 原图) 翻转后啥事都不干算一张新图

    AU_level = AU_repeat_level(math.ceil(len(AU_count) / (len(level_repeat))), AU_count)
    AU_level['0'] = 0  # 全都是背景那就是0
    print(AU_level)
    mix_database = dataset

    enhance_mix_database.update(mix_database)
    print("after deep copy")
    pool = mp.Pool(processes=mp.cpu_count())
    for idx,(img_path, AU_labels) in enumerate(mix_database.items()):
        orig_AU_labels = copy.copy(AU_labels)
        AU_labels = list(filter(lambda AU: not AU.startswith("?"), list(AU_labels)))
        if len(AU_labels) == 0:
            print("no AU occur! {}".format(img_path))
        max_repeat_times = level_repeat[max(AU_level[AU] for AU in AU_labels)]  # 计算每个图的label中需要重复最多的那个AU
        repeat_level = max(AU_level[AU] for AU in AU_labels)
        subject_name = img_path.split("/")[-3]
        sequence_name = img_path.split("/")[-2]
        frame_name = img_path.split("/")[-1]
        frame_name = frame_name[:frame_name.rindex(".")]
        new_path_lst = []
        need_generate = False #FIXME
        pool.apply_async(async_generate_image_save, args=(new_path_lst,
                                                           repeat_level,
                                                          subject_name, sequence_name, frame_name,
                                                          orig_AU_labels, img_path,need_generate
                                                          ),
                         callback=append_new_path_result)
    pool.close()
    pool.join()

    print("all process done, img_from:{}".format(len(img_from)))

    if not drop_big_label:
        return enhance_mix_database, img_from
    print("img enhance generate done, new_database:{}".format(len(enhance_mix_database)))
    AU_img_path = defaultdict(list)
    for img_path, AU_labels in enhance_mix_database.items():  # enhance
        AU_labels = filter(lambda AU: not AU.startswith("?"), list(AU_labels))
        for AU in AU_labels:
            AU_img_path[AU].append(img_path)

    picked_set = set()
    picked_AU_count = defaultdict(int)
    pick_count = 40000 # FIXME int(np.median(sorted([len(lst) for lst in AU_img_path.values()])))
    for AU, img_id_lst in sorted(AU_img_path.items(), key=lambda e: len(e[1])):
        # print(AU, len(img_id_lst))
        pick_set = set(img_id_lst) & picked_set
        remain_set = set(img_id_lst) - pick_set
        current_pick_count = min(pick_count, len(pick_set))
        remain_len = np.min([pick_count - current_pick_count, len(remain_set),
                             max(0, pick_count - picked_AU_count[AU])])  # 应该修改为看看历史上已选择AU有多少个了

        choice_array = np.array([])
        if len(pick_set) > 0:
            choice_array = np.random.choice(list(pick_set), current_pick_count, replace=False)  # 先挑选以前已经挑过的
        remain_array = np.random.choice(list(remain_set), remain_len, replace=False)  # 再挑补集
        # print("choice_array:{0}, remain_array:{1} add:{2}".format(len(choice_array), len(remain_array), len(choice_array)+ len(remain_array)))
        choice_array = np.hstack((choice_array, remain_array))
        picked_set.update(choice_array.tolist())

    new_enhance_dataset = defaultdict(set)
    for pick_img_path in picked_set:
        new_enhance_dataset[pick_img_path] = enhance_mix_database[pick_img_path]
    return new_enhance_dataset, img_from

# 下面这个函数解决分类不平衡问题的最佳方案,但会丢失大类的训练数据
def database_mixenhance_uniform_pick_check():
    # AU_count, img_idx_AU_DISFA = AU_stats_DISFA("/home/machen/dataset/DISFA/AU_labels/", skip_frame=1)

    AU_count_BP4D, img_idx_AU_BP4D = AU_stats_BP4D(config.DATA_PATH["BP4D"]+"/AUCoding/")
    # img_idx_AU_DISFA.update(img_idx_AU_BP4D) # mix
    #
    # for AU, count in AU_count_BP4D.items():
    #     AU_count[AU] += count  # mix add

    level_repeat = {0: 0,
                    1: 5,  # 锐化，模糊，原图, 翻转
                    2: 9, } # 左右翻转 x 锐化，模糊，原图 或 左右翻转 x 对比度增强，对比度降低，原图
                    #3: 10}  # 左右翻转 x (锐化，模糊，对比度增强，对比度降低, 原图)
    AU_level = AU_repeat_level(math.ceil(len(AU_count_BP4D) /(len(level_repeat))), AU_count_BP4D)
    print(AU_level)

    mix_database = img_idx_AU_BP4D
    enhance_mix_database = copy.deepcopy(mix_database)
    for img_id, AU_labels in mix_database.items():
        AU_labels = list(AU_labels)
        max_repeat_times = level_repeat[max(AU_level[AU] for AU in AU_labels)]  # 计算每个图的label中需要重复最多的那个AU
        for repeat_idx in range(max_repeat_times):
            enhance_mix_database["{0}_{1}".format(img_id, repeat_idx)].update(AU_labels) # enhance


    AU_imgid = defaultdict(list)
    for img_id, AU_labels in enhance_mix_database.items():  # enhance
        for AU in AU_labels:
            AU_imgid[AU].append(img_id)

    picked_set = set()
    picked_AU_count = defaultdict(int)
    pick_count =30000 # int(np.median(sorted([len(lst) for lst in AU_imgid.values()])))
    for AU, img_id_lst in sorted(AU_imgid.items(), key=lambda e:len(e[1])):
        # print(AU, len(img_id_lst))
        pick_set = set(img_id_lst) & picked_set
        remain_set = set(img_id_lst) - pick_set
        current_pick_count = min(pick_count, len(pick_set))
        remain_len = np.min([pick_count - current_pick_count, len(remain_set), max(0, pick_count - picked_AU_count[AU])]) # 应该修改为看看历史上已选择AU有多少个了

        choice_array = np.array([])
        if len(pick_set) > 0:
            choice_array = np.random.choice(list(pick_set), current_pick_count, replace=False)  # 先挑选以前已经挑过的
        remain_array = np.random.choice(list(remain_set), remain_len, replace=False) # 再挑补集
        # print("choice_array:{0}, remain_array:{1} add:{2}".format(len(choice_array), len(remain_array), len(choice_array)+ len(remain_array)))
        choice_array = np.hstack((choice_array, remain_array))
        for choice_img_id in choice_array:
            if choice_img_id not in picked_set:
                for enhance_AU_label in enhance_mix_database[choice_img_id]:
                    picked_AU_count[enhance_AU_label] += 1
        picked_set.update(choice_array.tolist())

    choice_AU_count = defaultdict(int)
    print("all choice count:{}".format(len(picked_set)))
    #stats again
    for img_id in picked_set:
        AU_labels = enhance_mix_database[img_id]
        for AU in AU_labels:
            choice_AU_count[AU] += 1

    print("remains lost count:{0}  interset count:{1}".format(len(set(list(mix_database.keys())) - picked_set),
                                                              len(set(list(mix_database.keys())) & picked_set)))
    return choice_AU_count




if __name__ == "__main__":
    stats_DISFA, _ = AU_stats_DISFA(config.DATA_PATH["DISFA"] + "/AU_labels/", skip_frame=1)


    stats_BP4D, img_idx_AU = AU_stats_BP4D(config.DATA_PATH["BP4D"]+"/AUCoding/")
    print("BP4D all len:{}".format(len(img_idx_AU)))
    # for AU, count in stats2.items():
    #     stats[AU] += count
    print("--------------------------------------------------------------")
    orig_first_count = list(sorted(stats_BP4D.items(), key=lambda e:e[1], reverse=True))[0][1]
    for AU, count in sorted(stats_BP4D.items(), key=lambda e:e[1], reverse=True):
        print("BP4D AU={0}, count={1}, ratio={2}".format(AU, count, orig_first_count/count))
    print("---------------------------------------------------")
    orig_first_count = list(sorted(stats_DISFA.items(), key=lambda e: e[1], reverse=True))[0][1]
    for AU, count in sorted(stats_DISFA.items(), key=lambda e: e[1], reverse=True):
        print("DISFA AU={0}, count={1}, ratio={2}".format(AU, count, orig_first_count / count))
    print("---------------------------------------------------")
    print("===================================")
    import config
    print("DISFA - BP4D", sorted(set(stats_DISFA.keys()) - set(stats_BP4D.keys())))
    print("BP4D - DISFA", set(stats_BP4D.keys()) - set(stats_DISFA.keys()))
    print("BP4D sorted AU: ", sorted(stats_BP4D.keys()))
    print("DISFA sorted AU: ", sorted(stats_DISFA.keys()))

    print("AU_ROI config & BP4D:", sorted(map(int , set(config.AU_ROI.keys()) & set(stats_BP4D.keys()))))
    print("AU_ROI config & DISFA:", sorted(map(int , set(config.AU_ROI.keys()) & set(stats_DISFA.keys()))))
    print("===================================")
    choice_AU_count = database_mixenhance_uniform_pick_check()
    first_count = list(sorted(choice_AU_count.items(), key=lambda e:e[1], reverse=True))[0][1]
    for AU, count in sorted(choice_AU_count.items(), key=lambda e:e[1], reverse=True):
        print("AU={0}, count={1}, ratio={2}".format(AU, count, first_count/count))
    '''BP4D dataset
    AU=10, count=87271
    AU=12, count=82531
    AU=7, count=80617
    AU=14, count=68376
    AU=6, count=67677
    AU=17, count=50407
    AU=1, count=31043
    AU=4, count=29755
    AU=2, count=25110
    AU=15, count=24869
    AU=23, count=24288
    AU=24, count=22229
    
    AU=9, count=8512
    AU=11, count=7184
    AU=16, count=6593
    AU=28, count=5697
    AU=5, count=5693
    AU=20, count=3644
    AU=27, count=1271
    AU=22, count=606
    AU=18, count=568
    AU=13, count=138
    '''
    '''
    BP4D and DISFA mix
    combine AU=12, count=113325
    combine AU=10, count=87271
    combine AU=6, count=87161
    combine AU=7, count=80617
    combine AU=14, count=68376
    combine AU=17, count=63337
    combine AU=4, count=54349
    combine AU=25, count=46052
    combine AU=1, count=39821
    combine AU=15, count=32731
    combine AU=2, count=32474
    combine AU=26, count=24976
    combine AU=23, count=24288
    combine AU=24, count=22229
    combine AU=9, count=15644
    combine AU=5, count=8422
    combine AU=20, count=8176
    combine AU=11, count=7184
    combine AU=16, count=6593
    combine AU=28, count=5697
    combine AU=27, count=1271
    combine AU=22, count=606
    combine AU=18, count=568
    combine AU=13, count=138
   '''
