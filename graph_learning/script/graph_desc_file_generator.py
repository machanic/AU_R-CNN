import sys
sys.path.insert(0, "/home/machen/face_expr")
import chainer
import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
from collections import defaultdict
import os
from dataset_toolkit.adaptive_AU_config import adaptive_AU_relation, adaptive_AU_database
import numpy as np
import cv2
import itertools
from collections import OrderedDict
from collections_toolkit.ordered_set import OrderedSet
from AU_rcnn.links.model.faster_rcnn import FasterRCNNResnet101
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_vgg import FasterRCNNVGG16
import argparse
from img_toolkit.face_mask_cropper import FaceMaskCropper
import re
import multiprocessing as mp

from functools import lru_cache
from operator import itemgetter
import itertools

def delegate_mask_crop(img_path, channal_first, queue):
    try:
        print("before crop {}".format(img_path))
        cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img_path, channal_first)
        print("crop {} done".format(img_path))
        queue.put((img_path, cropped_face, AU_mask_dict), block=True)
    except IndexError:
        pass

def read_DISFA_video_label(output_dir, is_binary_AU, is_need_adaptive_AU_relation=False, force_generate=True,
                           proc_num=10, cut=False, train_subject=None):
    mgr = mp.Manager()
    queue = mgr.Queue(maxsize=20000)
    for orientation in ["Left", "Right"]:
        if is_need_adaptive_AU_relation:
            adaptive_AU_relation()  # delete AU relation pair occur in same facial region
        au_couple_dict = get_zip_ROI_AU()
        au_couple_child_dict = get_AU_couple_child(au_couple_dict)
        DISFA_base_dir = config.DATA_PATH["DISFA"]
        label_file_dir = DISFA_base_dir + "/ActionUnit_Labels/"
        img_folder = DISFA_base_dir+ "/Img_{}Camera".format(orientation)
        for video_name in os.listdir(label_file_dir): # each file is a video
            is_train = True if video_name in train_subject else False
            if not force_generate:
                prefix = "train" if is_train else "test"
                target_file_path = output_dir + os.sep + prefix + os.sep + video_name+"_"+ orientation + ".npz"
                if os.path.exists(target_file_path):
                   continue
            resultdict={}
            if proc_num > 1:
                pool = mp.Pool(processes=proc_num)
                procs = 0
                one_file_name = os.listdir(label_file_dir + os.sep + video_name)[0]
                with open(label_file_dir + os.sep + video_name + os.sep + one_file_name, "r") as file_obj:
                    for idx, line in enumerate(file_obj):
                        line = line.strip()
                        if line:
                            frame = line.split(",")[0]
                            img_path = img_folder + "/{0}/{1}.jpg".format(video_name,frame)

                            pool.apply_async(func=delegate_mask_crop, args=(img_path, True, queue))
                            procs += 1
                for i in range(procs):
                    try:
                        entry = queue.get(block=True, timeout=60)
                        resultdict[entry[0]] = (entry[1], entry[2])
                    except Exception:
                        print("queue block time out")
                        break
                pool.close()
                pool.join()
                del pool
            else:  # only one process
                one_file_name = os.listdir(label_file_dir + os.sep + video_name)[0]
                with open(label_file_dir + os.sep + video_name + os.sep + one_file_name, "r") as file_obj:
                    for idx, line in enumerate(file_obj):
                        line = line.strip()
                        if line:
                            frame = line.split(",")[0]
                            img_path = img_folder + "/{0}/{1}.jpg".format(video_name,frame)
                            try:
                                cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img_path, True)
                                resultdict[img_path] = (cropped_face, AU_mask_dict)
                            except IndexError:
                                pass

            frame_label = dict()
            video_info = []
            video_img_path_set = set()
            for file_name in os.listdir(label_file_dir+os.sep + video_name):  # each file is one AU ( video file )
                AU = file_name[file_name.index("au")+2: file_name.rindex(".")]

                with open(label_file_dir+os.sep + video_name + os.sep + file_name, "r") as file_obj:
                    for line in file_obj:
                        frame = int(line.split(",")[0])
                        AU_intensity = int(line.split(",")[1])
                        img_path = img_folder + "/{0}/{1}.jpg".format(video_name, frame)
                        video_img_path_set.add((frame,img_path))
                        if frame not in frame_label:
                            frame_label[frame] = set()
                        if AU_intensity >= 1:  # NOTE that we apply AU_intensity >= 1
                            frame_label[int(frame)].add(AU)  #存储str类型
            for frame, img_path in sorted(video_img_path_set,key=lambda e:int(e[0])):
                if img_path not in resultdict:
                    continue
                AU_set = frame_label[frame]  # it is whole image's AU set

                if cut and len(AU_set) == 0:
                    continue
                cropped_face, AU_mask_dict = resultdict[img_path]


                all_couple_mask_dict = OrderedDict()
                for AU in sorted(map(int, config.AU_ROI.keys())):  # ensure same order
                    all_couple_mask_dict[au_couple_dict[str(AU)]] = AU_mask_dict[str(AU)]

                all_labels = list()  # 开始拼接all_labels
                for AU_couple in all_couple_mask_dict.keys():  # 顺序与all_couple_mask_dict一致
                    child_AU_couple_list = au_couple_child_dict[AU_couple]
                    AU_couple = set(AU_couple)
                    for child_AU_couple in child_AU_couple_list:
                        AU_couple.update(child_AU_couple)  # combine child region's AU
                    if not is_binary_AU:  # in CRF, CRF模式需要将同一个区域的多个AU用逗号分隔，拼接
                        concat_AU = []
                        for AU in AU_couple:
                            if AU in AU_set:  # AU_set 存储真实AU(ground truth label):str类型
                                concat_AU.append(AU)

                        if len(concat_AU) == 0:
                            all_labels.append("0")  # 若该区域压根没有任何AU出现，为了让只支持单label的CRF工作，用0来代替
                        else:
                            all_labels.append(",".join(sorted(concat_AU)))

                    else:  # convert to np.array which is AU_bin
                        AU_bin = np.zeros(len(config.AU_SQUEEZE)).astype(np.uint8)
                        for AU in AU_couple:
                            if AU in AU_set:  # judge if this region contain which subset of whole image's AU_set
                                np.put(AU_bin, config.AU_SQUEEZE.inv[AU], 1)
                        all_labels.append(tuple(AU_bin))

                video_info.append({"frame": frame, "cropped_face": cropped_face,
                                   "all_couple_mask_dict": all_couple_mask_dict, "all_labels": all_labels,
                                   "video_id": video_name+"_"+orientation})
            resultdict.clear()
            if video_info:
                yield video_info, video_name
            else:
                print("error in file:{} no video found".format(video_name+"_"+orientation))


def read_BP4D_video_label(output_dir, is_binary_AU, is_need_adaptive_AU_relation=False, force_generate=True,
                          proc_num=10, cut=False, train_subject=None):
    '''
    :param
            output_dir : 用于检查如果目标的文件已经存在，那么就不再生成
            is_binary_AU:
                          True --> return AU_binary 01010100
                          False --> used for CRF mode: single true AU label CRF/ or AU combination separate by comma
    :yield:  每个视频video收集齐了yield回去，视频中每一帧返回3部分：
            1. "img_path": /path/to/image
            1."all_couple_mask_dict": 是OrderedDict，包含所有区域的mask，不管AU是不是+1，还是-1(不管AU出现没出现)，key是AU_couple，来自于au_couple_dict = get_zip_ROI_AU()
            2."labels": 是list，index与all_couple_mask_dict一致，其中每个label
               要么是binary形式01010110，
               要么是3,4（由于一块位置可以发生多个AU，因此可以用逗号隔开的字符串来返回），根据is_binary_AU返回不同的值
    '''
    mgr = mp.Manager()
    queue = mgr.Queue(maxsize=20000)

    if is_need_adaptive_AU_relation:
        adaptive_AU_relation()  # delete AU relation pair occur in same facial region
    au_couple_dict = get_zip_ROI_AU()
    au_couple_child_dict = get_AU_couple_child(au_couple_dict)  # AU_couple => list of child AU_couple
    # if need_translate_combine_AU ==> "mask_path_dict":{(2,3,4): /pathtomask} convert to "mask_path_dict":{110: /pathtomask}
     # each is dict : {"img": /path/to/img, "mask_path_dict":{(2,3,4): /pathtomask}, }
    BP4D_base_dir_path = config.DATA_PATH["BP4D"]
    label_file_dir = BP4D_base_dir_path + "/AUCoding/"

    for file_name in os.listdir(label_file_dir):  # each file is a video

        subject_name = file_name[:file_name.index("_")]
        sequence_name = file_name[file_name.index("_") + 1: file_name.rindex(".")]
        is_train = True if subject_name in train_subject else False
        if not force_generate:
            prefix = "train" if is_train else "test"
            target_file_path = output_dir + os.sep + prefix + os.sep + subject_name + "_" + sequence_name + ".npz"
            if os.path.exists(target_file_path):
                continue
        resultdict = {}
        if proc_num > 1:

            one_image_path = os.listdir(config.TRAINING_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name)[0]
            zfill_len = len(one_image_path[:one_image_path.rindex(".")])

            procs = 0
            # read image file and crop and get AU mask
            pool = mp.Pool(processes=proc_num)
            with open(label_file_dir + "/" + file_name, "r") as au_file_obj:  # each file is a video
                for idx, line in enumerate(au_file_obj):

                    if idx == 0:
                        continue
                    lines = line.split(",")
                    frame = lines[0].zfill(zfill_len)

                    img_path = config.TRAINING_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name + os.sep + frame + ".jpg"
                    if not os.path.exists(img_path):
                        print("not exists img_path:{}".format(img_path))
                        continue

                    pool.apply_async(func=delegate_mask_crop, args=(img_path, True, queue))
                    procs += 1
                    # p = mp.Process(target=delegate_mask_crop, args=(img_path, True, queue))
                    # procs.append(p)
                    # p.start()

            for i in range(procs):
                try:
                    entry = queue.get(block=True, timeout=360)
                    resultdict[entry[0]] = (entry[1], entry[2])
                except Exception:
                    print("queue block time out")
                    break
            pool.close()
            pool.join()
            del pool
        else:  # only one process
            one_image_path = os.listdir(config.TRAINING_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name)[
                0]
            zfill_len = len(one_image_path[:one_image_path.rindex(".")])
            with open(label_file_dir + "/" + file_name, "r") as au_file_obj:  # each file is a video
                for idx, line in enumerate(au_file_obj):

                    lines = line.split(",")
                    frame = lines[0].zfill(zfill_len)

                    img_path = config.TRAINING_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name + os.sep + frame + ".jpg"
                    if not os.path.exists(img_path):
                        print("not exists img_path:{}".format(img_path))
                        continue
                    try:
                        cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img_path, channel_first=True)
                        # note that above AU_mask_dict, each AU may have mask that contains multiple separate regions
                        resultdict[img_path] = (cropped_face, AU_mask_dict)
                        print("one image :{} done".format(img_path))
                    except IndexError:
                        print("img_path:{} cannot obtain 68 landmark".format(img_path))
                        pass
        # for p in procs:
        #     p.join()
        AU_column_idx = {}
        with open(label_file_dir + "/" + file_name, "r") as au_file_obj:  # each file is a video

            video_info = []
            for idx, line in enumerate(au_file_obj):  # each line represent a frame image

                line = line.rstrip()
                lines = line.split(",")
                if idx == 0:  # header define which column is which Action Unit
                    for col_idx, AU in enumerate(lines[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue


                frame = lines[0].zfill(zfill_len)

                img_path = config.TRAINING_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name + os.sep + frame + ".jpg"
                if not os.path.exists(img_path):
                    print("not exists img_path:{}".format(img_path))
                    continue
                if img_path not in resultdict:
                    print("img_path:{} landmark not found, continue".format(img_path))
                    continue
                cropped_face, AU_mask_dict = resultdict[img_path]

                all_couple_mask_dict = OrderedDict()
                for AU in sorted(map(int, config.AU_ROI.keys())):  # ensure same order
                    all_couple_mask_dict[au_couple_dict[str(AU)]] = AU_mask_dict[str(AU)]


                au_label_dict = {AU: int(lines[AU_column_idx[AU]]) for AU in config.AU_ROI.keys()}  # store real AU label
                if cut and all(_au_label == 0 for _au_label in au_label_dict.values()):
                    continue
                all_labels = list()  # 开始拼接all_labels
                for AU_couple in all_couple_mask_dict.keys():  # 顺序与all_couple_mask_dict一致
                    child_AU_couple_list = au_couple_child_dict[AU_couple]
                    AU_couple = set(AU_couple)
                    for child_AU_couple in child_AU_couple_list:
                        AU_couple.update(child_AU_couple)  # label fetch: combine child region's AU
                    if not is_binary_AU: # in CRF, CRF模式需要将同一个区域的多个AU用逗号分隔，拼接
                        concat_AU = []
                        for AU in AU_couple:
                            if au_label_dict[AU] == 1:
                                concat_AU.append(AU)
                            elif au_label_dict[AU] == 9:
                                concat_AU.append("?{}".format(AU))

                        if len(concat_AU) == 0:
                            all_labels.append("0")  # 若该区域压根没有任何AU出现，为了让只支持单label的CRF工作，用0来代替
                        else:
                            all_labels.append(",".join(concat_AU))

                    else:  # convert to np.array which is AU_bin
                        AU_bin = np.zeros(len(config.AU_SQUEEZE)).astype(np.uint8)
                        for AU in AU_couple:
                            if au_label_dict[AU] == 9:
                                np.put(AU_bin, config.AU_SQUEEZE.inv[AU], -1)
                            elif au_label_dict[AU] == 1:
                                np.put(AU_bin, config.AU_SQUEEZE.inv[AU], 1)


                        all_labels.append(tuple(AU_bin))

                video_info.append({"frame": frame, "cropped_face": cropped_face,
                                   "all_couple_mask_dict":all_couple_mask_dict, "all_labels": all_labels,
                                   "video_id": subject_name + "_" + sequence_name})
        resultdict.clear()
        if video_info:
            yield video_info, subject_name
        else:
            print("error video_info:{}".format(file_name))

@lru_cache(maxsize=1024)
def has_edge(AU_couple_a, AU_couple_b, database):
    au_relation_set = None
    if database == "DISFA":
        au_relation_set = config.AU_RELATION_DISFA
    elif database == "BP4D":
        au_relation_set = config.AU_RELATION_BP4D
    for AU_a in AU_couple_a:
        for AU_b in AU_couple_b:
            possible_pair = tuple(sorted([int(AU_a), int(AU_b)]))
            if possible_pair in au_relation_set:
                return True
    return False

def build_graph(faster_rcnn, reader_func, output_dir, database_name, force_generate, proc_num, cut:bool, extract_key,
                train_subject, test_subject):
    '''
    currently CRF can only deal with single label situation
    so use /home/machen/dataset/BP4D/label_dict.txt to regard combine label as new single label
    example(each file contains one video!):
    node_id kown_label features
    1_12 +1 np_file:/path/to/npy features:1,3,4,5,5,...
    node_id specific: ${frame}_${roi}, eg: 1_12
    or
    444 +[0,0,0,1,0,1,0] np_file:/path/to/npy features:1,3,4,5,5,...
    spatio can have two factor node here, for example spatio_1 means upper face, and spatio_2 means lower face relation
    #edge 143 4289 spatio_1
    #edge 143 4289 spatio_2
    #edge 112 1392 temporal

    mode: RNN or CRF
    '''
    adaptive_AU_database(database_name)
    adaptive_AU_relation(database_name)

    is_binary_AU = True

    for video_info, subject_id in reader_func(output_dir, is_binary_AU=is_binary_AU, is_need_adaptive_AU_relation=False,
                                  force_generate=force_generate, proc_num=proc_num, cut=cut, train_subject=train_subject):

        node_list = []
        temporal_edges = []
        spatio_edges = []
        h_info_array = []
        box_geometry_array = []
        for entry_dict in video_info:
            frame = entry_dict["frame"]
            cropped_face = entry_dict["cropped_face"]
            print("processing frame:{}".format(frame))
            all_couple_mask_dict = entry_dict["all_couple_mask_dict"]  # key is AU couple tuple,不管脸上有没有该AU都返回回来
            image_labels = entry_dict["all_labels"]  # each region has a label(binary or AU)

            bboxes = []
            labels = []
            AU_couple_bbox_dict = dict()

            for idx, (AU_couple, mask) in enumerate(all_couple_mask_dict.items()):  # AU may contain single_true AU or AU binary tuple (depends on need_adaptive_AU_relation)
                region_label = image_labels[idx] # str or tuple, so all_labels index must be the same as all_couple_mask_dict
                connect_arr = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
                component_num = connect_arr[0]
                label_matrix = connect_arr[1]
                temp_boxes = []
                for component_label in range(1, component_num):
                    row_col = list(zip(*np.where(label_matrix == component_label)))
                    row_col = np.array(row_col)
                    y_min_index = np.argmin(row_col[:, 0])
                    y_min = row_col[y_min_index, 0]
                    x_min_index = np.argmin(row_col[:, 1])
                    x_min = row_col[x_min_index, 1]
                    y_max_index = np.argmax(row_col[:, 0])
                    y_max = row_col[y_max_index, 0]
                    x_max_index = np.argmax(row_col[:, 1])
                    x_max = row_col[x_max_index, 1]
                    # same region may be shared by different AU, we must deal with it
                    coordinates = (y_min, x_min, y_max, x_max)

                    if y_min == y_max and x_min == x_max:
                        continue
                    temp_boxes.append(coordinates)
                temp_boxes = sorted(temp_boxes, key=itemgetter(3)) # must make sure each frame have same box order
                for coordinates in temp_boxes:
                    if coordinates not in bboxes:
                        bboxes.append(coordinates)
                        labels.append(region_label)  # AU may contain single_true AU or AU binary tuple (depends on need_adaptive_AU_relation)
                        AU_couple_bbox_dict[coordinates] = AU_couple
                del label_matrix
            if len(bboxes) != config.BOX_NUM[database_name]:
                print("boxes num != {0}, real box num= {1}".format(config.BOX_NUM[database_name], len(bboxes)))
                continue
            with chainer.no_backprop_mode(),chainer.using_config('train', False):
                bboxes = np.asarray(bboxes, dtype=np.float32)
                h = faster_rcnn.extract(cropped_face, bboxes, layer=extract_key)  # shape = R' x 2048
            assert h.shape[0] == len(bboxes)
            h = chainer.cuda.to_cpu(h)
            h = h.reshape(len(bboxes), -1)

            # 这个indent级别都是同一张图片内部
            # print("box number, all_mask:", len(bboxes),len(all_couple_mask_dict))
            for box_idx, box in enumerate(bboxes):
                label = labels[box_idx]  # label maybe single true AU or AU binary tuple
                if isinstance(label, tuple):
                    label_arr = np.char.mod("%d", label)
                    label = "({})".format(",".join(label_arr))
                h_flat = h[box_idx]
                # nonzero_idx = np.nonzero(h_flat)[0]
                # h_flat_nonzero = h_flat[nonzero_idx]
                # h_info = ",".join("{}:{:.4f}".format(idx, val) for idx,val in zip(nonzero_idx,h_flat_nonzero))

                node_id = "{0}_{1}".format(frame, box_idx)
                node_list.append("{0} {1} feature_idx:{2}".format(node_id, label, len(h_info_array)))
                h_info_array.append(h_flat)
                box_geometry_array.append(box)

            # 同一张画面两两组合，看有没连接线，注意AU=0，就是未出现的AU动作的区域也参与连接
            for box_idx_a, box_idx_b in map(sorted, itertools.combinations(range(len(bboxes)), 2)):
                node_id_a = "{0}_{1}".format(frame, box_idx_a)
                node_id_b = "{0}_{1}".format(frame, box_idx_b)
                AU_couple_a = AU_couple_bbox_dict[bboxes[box_idx_a]]  # AU couple represent region( maybe symmetry in face)
                AU_couple_b = AU_couple_bbox_dict[bboxes[box_idx_b]]
                if AU_couple_a == AU_couple_b or has_edge(AU_couple_a, AU_couple_b, database_name):
                    spatio_edges.append("#edge {0} {1} spatio".format(node_id_a, node_id_b))

        box_id_temporal_dict = defaultdict(list)  # key = roi/bbox id, value = node_id list cross temporal
        for node_info in node_list:
            node_id = node_info[0: node_info.index(" ")]
            box_id = node_id[node_id.index("_")+1:]
            box_id_temporal_dict[box_id].append(node_id)

        for node_id_list in box_id_temporal_dict.values():
            for idx, node_id in enumerate(node_id_list):
                if idx + 1 < len(node_id_list):
                    node_id_next = node_id_list[idx+1]
                    temporal_edges.append("#edge {0} {1} temporal".format(node_id, node_id_next))

        if subject_id in train_subject:
            output_path = "{0}/train/{1}.txt".format(output_dir, video_info[0]["video_id"])
        elif subject_id in test_subject:
            output_path = "{0}/test/{1}.txt".format(output_dir, video_info[0]["video_id"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        npz_path = output_path[:output_path.rindex(".")] + ".npz"

        np.savez(npz_path, appearance_features=np.asarray(h_info_array,dtype=np.float32),
                 geometry_features=np.array(box_geometry_array, dtype=np.float32))
        with open(output_path, "w") as file_obj:
            for line in node_list:
                file_obj.write("{}\n".format(line))
            for line in spatio_edges:
                file_obj.write("{}\n".format(line))
            for line in temporal_edges:
                file_obj.write("{}\n".format(line))
            file_obj.flush()
            node_list.clear()
            spatio_edges.clear()
            temporal_edges.clear()
            h_info_array.clear()
            box_geometry_array.clear()


def build_graph_roi_single_label(faster_rcnn, reader_func, output_dir, database_name, force_generate, proc_num, cut:bool, extract_key, train_subject, test_subject):
    '''
    currently CRF can only deal with single label situation
    so use /home/machen/dataset/BP4D/label_dict.txt to regard combine label as new single label
    example(each file contains one video!):
    node_id kown_label features
    1_12 +1 np_file:/path/to/npy features:1,3,4,5,5,...
    node_id specific: ${frame}_${roi}, eg: 1_12
    or
    444 +[0,0,0,1,0,1,0] np_file:/path/to/npy features:1,3,4,5,5,...
    spatio can have two factor node here, for example spatio_1 means upper face, and spatio_2 means lower face relation
    #edge 143 4289 spatio_1
    #edge 143 4289 spatio_2
    #edge 112 1392 temporal

    mode: RNN or CRF
    '''
    adaptive_AU_database(database_name)
    adaptive_AU_relation(database_name)
    au_couple_dict = get_zip_ROI_AU()  # value is AU couple tuple, each tuple denotes an RoI
    # max_au_couple_len = max(len(couple) for couple in au_couple_dict.values())  # we use itertools.product instead
    label_bin_len = config.BOX_NUM[database_name]  # each box/ROI only have 1 or 0
    au_couple_set = set(au_couple_dict.values())
    au_couple_list = list(au_couple_set)
    au_couple_list.append(("1","2","5","7")) # because it is symmetric area
    is_binary_AU = True

    for video_info, subject_id in reader_func(output_dir, is_binary_AU=is_binary_AU, is_need_adaptive_AU_relation=False,
                                  force_generate=force_generate, proc_num=proc_num, cut=cut, train_subject=train_subject):

        extracted_feature_cache = dict()  # key = np.ndarray_hash , value = h. speed up
        frame_box_cache = dict()  # key = frame, value = boxes
        frame_labels_cache = dict()
        frame_AU_couple_bbox_dict_cache = dict()
        # each video file is copying multiple version but differ in label
        if database_name == "BP4D":
            label_split_list = config.BP4D_LABEL_SPLIT
        elif database_name == "DISFA":
            label_split_list = config.DISFA_LABEL_SPLIT
        for couples_tuple in label_split_list:  # couples_tuple = ("1","3","5",.."4") cross AU_couple, config.LABEL_SPLIT come from frequent pattern statistics
            assert len(couples_tuple) == config.BOX_NUM[database_name]
            couples_tuple = tuple(map(str, sorted(map(int, couples_tuple))))
            couples_tuple_set = set(couples_tuple)  # use cartesian product to iterator over
            if len(couples_tuple_set) < len(couples_tuple):
                continue
            # limit too many combination
            # count = 0
            # for fp in fp_set:
            #     inter_set = couples_tuple_set & set(fp)
            #     union_set = couples_tuple_set | set(fp)
            #     iou = len(inter_set) / len(union_set)
            #     if iou > 0.6:
            #         count += 1
            # if count < 20:
            #     continue

            node_list = []
            temporal_edges = []
            spatio_edges = []
            h_info_array = []
            box_geometry_array = []
            for entry_dict in video_info:
                frame = entry_dict["frame"]
                cropped_face = entry_dict["cropped_face"]
                print("processing frame:{}".format(frame))
                all_couple_mask_dict = entry_dict["all_couple_mask_dict"]  # key is AU couple tuple,不管脸上有没有该AU都返回回来
                image_labels = entry_dict["all_labels"]  # each region has a label(binary or AU)


                bboxes = []
                labels = []
                AU_couple_bbox_dict = OrderedDict()


                if frame in frame_box_cache:
                    bboxes = frame_box_cache[frame]
                    labels = frame_labels_cache[frame]
                    AU_couple_bbox_dict = frame_AU_couple_bbox_dict_cache[frame]
                else:

                    for idx, (AU_couple, mask) in enumerate(all_couple_mask_dict.items()):  # We cannot sort this dict here, because region_label depend on order of this dict.AU may contain single_true AU or AU binary tuple (depends on need_adaptive_AU_relation)
                        region_label = image_labels[idx]  # str or tuple, so all_labels index must be the same as all_couple_mask_dict
                        connect_arr = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
                        component_num = connect_arr[0]
                        label_matrix = connect_arr[1]
                        for component_label in range(1, component_num):
                            row_col = list(zip(*np.where(label_matrix == component_label)))
                            row_col = np.array(row_col)
                            y_min_index = np.argmin(row_col[:, 0])
                            y_min = row_col[y_min_index, 0]
                            x_min_index = np.argmin(row_col[:, 1])
                            x_min = row_col[x_min_index, 1]
                            y_max_index = np.argmax(row_col[:, 0])
                            y_max = row_col[y_max_index, 0]
                            x_max_index = np.argmax(row_col[:, 1])
                            x_max = row_col[x_max_index, 1]
                            # same region may be shared by different AU, we must deal with it
                            coordinates = (y_min, x_min, y_max, x_max)

                            if y_min == y_max and x_min == x_max:
                                continue

                            if coordinates not in bboxes:
                                bboxes.append(coordinates)  # bboxes and labels have the same order
                                labels.append(region_label)  # AU may contain single_true AU or AU binary tuple (depends on need_adaptive_AU_relation)
                                AU_couple_bbox_dict[coordinates] = AU_couple

                        del label_matrix
                    if len(bboxes) != config.BOX_NUM[database_name]:
                        print("boxes num != {0}, real box num= {1}".format(config.BOX_NUM[database_name], len(bboxes)))
                        continue
                frame_box_cache[frame] = bboxes
                frame_AU_couple_bbox_dict_cache[frame] = AU_couple_bbox_dict
                frame_labels_cache[frame] = labels
                box_idx_AU_dict = dict()  # box_idx => AU, cannot cache! because couples_tuple each time is different
                already_added_AU_set = set()
                for box_idx, _ in enumerate(bboxes):  # bboxes may from cache
                    AU_couple = list(AU_couple_bbox_dict.values())[box_idx]  # AU_couple_bbox_dict may from cache
                    for AU in couples_tuple:  # couples_tuple not from cache, thus change after each iteration 每轮迭代完的时候变换
                        if AU in AU_couple and AU not in already_added_AU_set:
                            box_idx_AU_dict[box_idx] = (AU, AU_couple)
                            already_added_AU_set.add(AU)
                            break

                cropped_face.flags.writeable = False
                key = hash(cropped_face.data.tobytes())
                if key in extracted_feature_cache:
                    h = extracted_feature_cache[key]
                else:
                    with chainer.no_backprop_mode(),chainer.using_config('train', False):
                        h = faster_rcnn.extract(cropped_face, bboxes, layer=extract_key)  # shape = R' x 2048
                        extracted_feature_cache[key] = h
                    assert h.shape[0] == len(bboxes)
                h = chainer.cuda.to_cpu(h)
                h = h.reshape(len(bboxes), -1)

                # 这个indent级别都是同一张图片内部
                # print("box number, all_mask:", len(bboxes),len(all_couple_mask_dict))
                assert len(box_idx_AU_dict) == config.BOX_NUM[database_name]
                for box_idx, (AU, AU_couple) in sorted(box_idx_AU_dict.items(), key=lambda e: int(e[0])):
                    label = np.zeros(shape=label_bin_len, dtype=np.int32)  # bin length became box number > AU_couple number
                    AU_squeeze_idx = config.AU_SQUEEZE.inv[AU]
                    label[couples_tuple.index(AU)] = labels[box_idx][AU_squeeze_idx]  # labels缓存起来可能出错 # labels[box_idx] = 0,0,1,1,...,0  but we want only look at specific idx
                    label = tuple(label)
                    label_arr = np.char.mod("%d", label)
                    label = "({})".format(",".join(label_arr))
                    h_flat = h[box_idx]                     
                    node_id = "{0}_{1}".format(frame, box_idx)
                    node_list.append("{0} {1} feature_idx:{2} AU_couple:{3} AU:{4}".format(node_id, label, len(h_info_array), AU_couple, AU))
                    h_info_array.append(h_flat)
                    box_geometry_array.append(bboxes[box_idx])

                # 同一张画面两两组合，看有没连接线，注意AU=0，就是未出现的AU动作的区域也参与连接
                for box_idx_a, box_idx_b in map(sorted, itertools.combinations(range(len(bboxes)), 2)):
                    node_id_a = "{0}_{1}".format(frame, box_idx_a)
                    node_id_b = "{0}_{1}".format(frame, box_idx_b)
                    AU_couple_a = AU_couple_bbox_dict[bboxes[box_idx_a]]  # AU couple represent region( maybe symmetry in face)
                    AU_couple_b = AU_couple_bbox_dict[bboxes[box_idx_b]]
                    if AU_couple_a == AU_couple_b or has_edge(AU_couple_a, AU_couple_b, database_name):
                        spatio_edges.append("#edge {0} {1} spatio".format(node_id_a, node_id_b))

            box_id_temporal_dict = defaultdict(list)  # key = roi/bbox id, value = node_id list cross temporal
            for node_info in node_list:
                node_id = node_info[0: node_info.index(" ")]
                box_id = node_id[node_id.index("_")+1:]
                box_id_temporal_dict[box_id].append(node_id)

            for node_id_list in box_id_temporal_dict.values():
                for idx, node_id in enumerate(node_id_list):
                    if idx + 1 < len(node_id_list):
                        node_id_next = node_id_list[idx+1]
                        temporal_edges.append("#edge {0} {1} temporal".format(node_id, node_id_next))
            train_AU_out_path = "{0}/train/{1}/{2}.txt".format(output_dir, "_".join(map(str, couples_tuple)),
                                                               video_info[0]["video_id"] )
            test_AU_out_path = "{0}/test/{1}/{2}.txt".format(output_dir, "_".join(map(str,  couples_tuple)),
                                                             video_info[0]["video_id"] )
            if subject_id in train_subject:
                output_path = train_AU_out_path
                npz_path = output_dir + os.sep + "train" + os.sep + os.path.basename(output_path)[
                                                                    :os.path.basename(output_path).rindex(".")] + ".npz"
            elif subject_id in test_subject:
                output_path = test_AU_out_path
                npz_path = output_dir + os.sep + "test" + os.sep + os.path.basename(output_path)[
                                                                    :os.path.basename(output_path).rindex(".")] + ".npz"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if not os.path.exists(npz_path):
                np.savez(npz_path, appearance_features=h_info_array,
                         geometry_features=np.array(box_geometry_array,dtype=np.float32))
            with open(output_path, "w") as file_obj:
                for line in node_list:
                    file_obj.write("{}\n".format(line))
                for line in spatio_edges:
                    file_obj.write("{}\n".format(line))
                for line in temporal_edges:
                    file_obj.write("{}\n".format(line))
                file_obj.flush()
                node_list.clear()
                spatio_edges.clear()
                temporal_edges.clear()
                h_info_array.clear()





def load_train_test_id(folder_path, split_idx, database):
    train_subject_id_set = set()
    test_subject_id_set = set()
    with open(folder_path + os.sep + "id_trainval_{}.txt".format(split_idx), "r") as file_obj:
        for line in file_obj:
            if database == "BP4D":
                subject_id = line[:line.index("/")]
                train_subject_id_set.add(subject_id)
            elif database == "DISFA":
                subject_id = line.split("/")[1]
                train_subject_id_set.add(subject_id)
    with open(folder_path + os.sep + "id_test_{}.txt".format(split_idx), "r") as file_obj:
        for line in file_obj:
            if database == "BP4D":
                subject_id = line[:line.index("/")]
                test_subject_id_set.add(subject_id)
            elif database == "DISFA":
                subject_id = line.split("/")[1]
                test_subject_id_set.add(subject_id)
    return train_subject_id_set, test_subject_id_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate Graph desc file script')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_no_enhance.npy", help='image mean .npy file')
    parser.add_argument("--output", default="/home/machen/face_expr/result/graph")
    parser.add_argument("--model", default="/home/machen/face_expr/result/10_fold_5_resnet101_linear_snapshot_model.npz")
    parser.add_argument("--prefix", default='',help="can be _pre")
    parser.add_argument("--pretrained_model_name", '-premodel', default='resnet101')
    parser.add_argument("--proc_num","-proc", type=int,default=10)
    parser.add_argument('--database', default='BP4D',
                        help='Output directory')
    parser.add_argument('--device', default=1, type=int,
                        help='GPU device number')
    parser.add_argument('--use_lstm', action='store_true', help='use LSTM or Linear in head module')
    parser.add_argument('--extract_len', type=int, default=1000)
    parser.add_argument("--cut_zero", '-cut', action="store_true")
    parser.add_argument("--roi_label_split",  action="store_true",
                        help="use 'roi label split' strategy to choose only one "
                             "single label to help to refine label in video sequence more reasonable")

    chainer.config.train = False
    args = parser.parse_args()
    kfold_pattern = re.compile('.*?(\d+)_.*?fold_(\d+).*',re.DOTALL)
    matcher = kfold_pattern.match(args.model)

    if matcher:
        fold = matcher.group(1)
        split_idx = matcher.group(2)
    output = args.output
    if args.prefix:
        id_list_fold_path = config.DATA_PATH[args.database] + "/idx/{0}_fold{1}/".format(fold, args.prefix)
    else:
        id_list_fold_path = config.DATA_PATH[args.database] + "/idx/{0}_fold/".format(fold)
    train_subject, test_subject = load_train_test_id(id_list_fold_path, split_idx, args.database)
    os.makedirs(output, exist_ok=True)

    adaptive_AU_database(args.database)
    extract_key = ""

    if args.pretrained_model_name == "resnet101":
        faster_rcnn = FasterRCNNResnet101(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model="resnet101",
                                      mean_file=args.mean, use_lstm=args.use_lstm, extract_len=args.extract_len)  # 可改为/home/machen/face_expr/result/snapshot_model.npz
        extract_key = 'avg_pool'
    elif args.pretrained_model_name == "vgg":
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model="imagenet",
                                      mean_file=args.mean,
                                      use_lstm=False,
                                      extract_len=args.extract_len)
        extract_key = 'fc'
    if os.path.exists(args.model):
        print("loading pretrained snapshot:{}".format(args.model))
        chainer.serializers.load_npz(args.model, faster_rcnn)
    else:
        print("error, not exists pretrained model file:{}".format(args.model))
        sys.exit(1)

    if args.device >= 0:
        faster_rcnn.to_gpu(args.device)
        chainer.cuda.get_device_from_id(int(args.device)).use()

    # print("GPU load done")
    if args.database == "BP4D":
        read_func = read_BP4D_video_label
    elif args.database == "DISFA":
        read_func = read_DISFA_video_label
    else:
        print("you can not specify database other than BP4D/DISFA")
        sys.exit(1)
    if args.roi_label_split:
        build_graph_roi_single_label(faster_rcnn, read_func, output, database_name=args.database,
                                     force_generate=False,
                                     proc_num=args.proc_num, cut=args.cut_zero, extract_key=extract_key,
                                     train_subject=train_subject,
                                     test_subject=test_subject)
    else:
        build_graph(faster_rcnn, read_func, output, database_name=args.database, force_generate=False,
                    proc_num=args.proc_num, cut=args.cut_zero, extract_key=extract_key, train_subject=train_subject,
                    test_subject=test_subject)









