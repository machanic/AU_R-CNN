import sys
sys.path.insert(0, "/home/machen/face_expr")
import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU
from collections import defaultdict
import os
from dataset_toolkit.adaptive_AU_config import adaptive_AU_relation, adaptive_AU_database
from AU_rcnn.utils import read_image
import numpy as np
import cv2
import itertools
from collections import OrderedDict
from collections_toolkit.ordered_set import OrderedSet
from AU_rcnn.links import FasterRCNNVGG16
import argparse
import glob

def read_BP4D_video_bbox(output_dir, is_binary_AU, is_need_adaptive_AU_relation=False, force_generate=True):
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
    if is_need_adaptive_AU_relation:
        adaptive_AU_relation() # delete AU relation pair occur in same facial region
    BP4D_base_dir_path = config.DATA_PATH["BP4D"]
    label_file_dir = BP4D_base_dir_path + "/AUCoding/"
    au_mask_dir = BP4D_base_dir_path + "/BP4D_AUmask/"
    au_couple_dict = get_zip_ROI_AU()
    # if need_translate_combine_AU ==> "mask_path_dict":{(2,3,4): /pathtomask} convert to "mask_path_dict":{110: /pathtomask}
     # each is dict : {"img": /path/to/img, "mask_path_dict":{(2,3,4): /pathtomask}, }
    for file_name in os.listdir(label_file_dir): # each file is a video
        video_info = []
        subject_name = file_name[:file_name.index("_")]
        sequence_name = file_name[file_name.index("_") + 1: file_name.rindex(".")]
        if not force_generate:
            target_file_path = output_dir + os.sep + subject_name + "_" + sequence_name + ".txt"
            if os.path.exists(target_file_path):
                continue
        one_image_path = os.listdir(config.CROP_DATA_PATH["BP4D"] + os.sep + subject_name + os.sep + sequence_name)[0]
        zfill_len = len(one_image_path[:one_image_path.rindex(".")])
        AU_column_idx = {}

        with open(label_file_dir + "/" + file_name, "r") as au_file_obj:  # each file is a video

            for idx, line in enumerate(au_file_obj):  # each line represent a frame image

                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue

                lines = line.split(",")
                frame = lines[0].zfill(zfill_len)

                img_path = config.CROP_DATA_PATH["BP4D"] + os.sep + subject_name+os.sep + sequence_name + os.sep + frame + ".jpg"
                if not os.path.exists(img_path):
                    print("not exists img_path:{}".format(img_path))
                    continue

                au_couple_labels = OrderedSet()
                for AU in sorted(map(int, config.AU_ROI.keys())):
                    au_couple_labels.add(au_couple_dict[str(AU)])
                all_couple_mask_dict = OrderedDict()  # 用OrderedDict可以保证与labels同样的顺序
                for AU_couple in au_couple_labels:
                    all_couple_mask_dict[AU_couple] = "{0}/{1}/{2}/{3}_AU_{4}.png".format(au_mask_dir,
                                                                                           subject_name, sequence_name,
                                                                                           frame,
                                                                                           ",".join(AU_couple))

                au_label_dict = {AU: int(lines[AU_column_idx[AU]]) for AU in config.AU_ROI.keys()}

                all_labels = list()
                for AU_couple, mask_path in all_couple_mask_dict.items():
                    if not is_binary_AU: # in CRF, CRF模式需要将同一个区域的多个AU用逗号分隔，拼接
                        concat_AU = []
                        for AU in AU_couple:
                            if au_label_dict[AU] == 1:
                                concat_AU.append(AU)
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

                video_info.append({"frame":frame, "img_path": img_path,
                                   "all_couple_mask_dict":all_couple_mask_dict, "all_labels":all_labels})

        if video_info:
            yield video_info
        else:
            print("error video_info:{}".format(file_name))

def has_edge(AU_couple_a, AU_couple_b):
    for AU_a in AU_couple_a:
        for AU_b in AU_couple_b:
            possible_pair = tuple(sorted([int(AU_a), int(AU_b)]))
            if possible_pair in config.AU_RELATION_BP4D:
                return True
    return False

def build_graph(faster_rcnn, reader_func, output_dir, database_name, mode, force_generate):
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
    adaptive_AU_relation()


    is_binary_AU = True if mode == "RNN" else False

    for video_info in reader_func(output_dir, is_binary_AU=is_binary_AU, is_need_adaptive_AU_relation=False,
                                  force_generate=force_generate):

        node_list = []
        temporal_edges = []  # 每个node用直线连接起来，这意味着对称区域（比如双眼，各自拥有一根线）
        spatio_edges = []

        for entry_dict in video_info:
            frame = entry_dict["frame"]
            img_path = entry_dict["img_path"]
            all_couple_mask_dict = entry_dict["all_couple_mask_dict"]  # key is AU couple tuple,不管脸上有没有该AU都返回回来
            all_labels = entry_dict["all_labels"]  # each region has a label(binary or AU)
            print("processing frame: {0}, path:{1}".format(frame, img_path))
            img = read_image(img_path, color=True)
            img = faster_rcnn.prepare(img)
            bboxes = []
            labels = []
            AU_couple_bbox_dict = dict()

            for idx, (AU_couple, mask_path) in enumerate(all_couple_mask_dict.items()):  # AU may contain single_true AU or AU binary tuple (depends on need_adaptive_AU_relation)
                region_label = all_labels[idx] # str or tuple
                # FIXME 不得已为之的办法，更好的办法是直接计算得到mask，mask_path on the fly
                mask_path_dir = os.path.dirname(mask_path)
                AU_mask_path_dict = dict()
                all_frame_mask_path_ls = glob.glob("{0}/{1}_AU_*".format(mask_path_dir, frame))

                for each_path in all_frame_mask_path_ls:
                    AU_ls = each_path[each_path.rindex("_") + 1:each_path.rindex(".")].split(",")
                    for _AU in AU_ls:
                        AU_mask_path_dict[_AU] = each_path
                orig_mask_path = mask_path
                AU_ls = mask_path[mask_path.rindex("_")+1:mask_path.rindex(".")].split(",")
                mask_path = AU_mask_path_dict[AU_ls[0]]
                if not os.path.exists(mask_path):
                    print("error mask:{}".format(orig_mask_path))
                    continue
                mask = read_image(mask_path, dtype=np.uint8, color=False)
                connect_arr = cv2.connectedComponents(mask[0], connectivity=4, ltype=cv2.CV_32S)
                component_num = connect_arr[0]
                label_matrix = connect_arr[1]

                actual_connect = 0
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
                    actual_connect += 1
                    if coordinates not in bboxes:
                        bboxes.append(coordinates)
                        labels.append(region_label)  # AU may contain single_true AU or AU binary tuple (depends on need_adaptive_AU_relation)
                        AU_couple_bbox_dict[coordinates] = AU_couple
                del label_matrix
                # print("AU_couple:,", AU_couple, "connect:", actual_connect, "mask:", mask_path)
            if len(bboxes) == 0:
                print("no box! err img:{}".format(img_path))
                continue
            h = faster_rcnn.extract(img, bboxes)  # shape = R' x 4096
            assert h.shape[0] == len(bboxes)
            # 这个indent级别都是同一张图片内部
            # print("box number, all_mask:", len(bboxes),len(all_couple_mask_dict))
            for box_idx in range(len(bboxes)):
                label = labels[box_idx]  # label maybe single true AU or AU binary tuple
                if isinstance(label, tuple):
                    label_arr = np.char.mod("%d", label)
                    label = "({})".format(",".join(label_arr))
                h_info = ",".join(map(str, h[box_idx, :]))  # feature become separate by comma
                node_id = "{0}_{1}".format(frame, box_idx)
                node_list.append("{0} {1} features:{2}".format(node_id, label, h_info))

            # 同一张画面两两组合，看有没连接线，注意AU=0，就是未出现的AU动作的区域也参与连接
            for box_idx_a, box_idx_b in map(sorted, itertools.combinations(range(len(bboxes)), 2)):
                node_id_a = "{0}_{1}".format(frame, box_idx_a)
                node_id_b = "{0}_{1}".format(frame, box_idx_b)
                AU_couple_a = AU_couple_bbox_dict[bboxes[box_idx_a]]  # AU couple represent region( maybe symmetry in face)
                AU_couple_b = AU_couple_bbox_dict[bboxes[box_idx_b]]
                if AU_couple_a == AU_couple_b:
                    spatio_edges.append("#edge {0} {1} symmetry".format(node_id_a, node_id_b))
                elif has_edge(AU_couple_a, AU_couple_b):
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

        output_path = "{0}/{1}_{2}.txt".format(output_dir,video_info[0]["img_path"].split(os.sep)[-3],
                                               video_info[0]["img_path"].split(os.sep)[-2])

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate Graph desc file script')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_enhance.npy", help='image mean .npy file')
    parser.add_argument("--output", default="/home/machen/face_expr/result/graph_backup")
    parser.add_argument('--database', default='BP4D',
                        help='Output directory')
    args = parser.parse_args()
    adaptive_AU_database(args.database)
    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                  pretrained_model="/home/machen/face_expr/result/snapshot_model.npz",
                                  mean_file=args.mean)  # 可改为/home/machen/face_expr/result/snapshot_model.npz
    build_graph(faster_rcnn, read_BP4D_video_bbox, args.output, database_name="BP4D", mode="RNN",force_generate=True)








