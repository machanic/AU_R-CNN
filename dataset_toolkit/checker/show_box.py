import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/machen/face_expr')
from AU_rcnn.utils import read_image
from AU_rcnn.utils.bbox.bbox_iou import bbox_iou, bbox_intersection_area
import random
import os
from dataset_toolkit.compress_utils import get_zip_ROI_AU
import config
from chainer import cuda
from collections_toolkit.ordered_set import OrderedSet


def merge_box(bbox, labels, threshold):
    '''

    :param bbox: shape (N, 4), N is number of box inside one image
    :param label: the same shape as bbox, each box just have one label, such as 3, 6, and so on
    :return: merged_bbox, merged_label, each box now has multi_label, so each label is 10010101 binary format
    '''
    xp = cuda.get_array_module(bbox)
    bbox = cuda.to_cpu(bbox)
    labels = cuda.to_cpu(labels)
    cal_area = lambda y_min, x_min, y_max, x_max: (y_max - y_min) * (x_max - x_min)
    binary_labels = []
    for label in labels:
        label = int(label)
        label_bin = np.zeros(len(config.AU_ROI) + 1, dtype=np.uint8)  # +1 means background label = 0
        np.put(label_bin, config.AU_SQUEEZE.inv[label], 1)
        binary_labels.append(label_bin)

    iou = bbox_iou(bbox, bbox)
    np.fill_diagonal(iou, 0)
    iou[iou < threshold] = 0
    iou[iou > 0] = 1
    iou = iou.astype(np.uint8)
    connect_arr = cv2.connectedComponents(iou, connectivity=4, ltype=cv2.CV_32S)

    label_matrix = connect_arr[1]
    # np.savetxt("/home/machen/orig-label.txt", label_matrix, fmt="%d", delimiter=' ')
    # need modify label_matrix, skip component in same row/col should also connect
    for row in label_matrix:
        if len(row[row > 0]) > 0:
            reference_label = np.min(row[row > 0])
            for each_component in row[row > 0]:
                label_matrix[label_matrix == each_component] = reference_label

    label_matrix = np.transpose(label_matrix)
    for row in label_matrix:
        if len(row[row > 0]) > 0:
            reference_label = np.min(row[row > 0])
            for each_component in row[row > 0]:
                label_matrix[label_matrix == each_component] = reference_label

    # np.savetxt("/home/machen/refined-label.txt", label_matrix, fmt="%d", delimiter=' ')
    all_non_zero_elements = set()
    for row, col in zip(*np.nonzero(label_matrix)):
        all_non_zero_elements.add(label_matrix[row, col])
    for idx, element in enumerate(sorted(list(all_non_zero_elements))):
        label_matrix[label_matrix == element] = idx+1
    np.savetxt("/home/machen/2nd-refined-label.txt", label_matrix, fmt="%d", delimiter=' ')
    component_num = np.max(label_matrix) + 1
    merged_bbox = []
    merged_label = []
    already_idx_set = set()
    for component_label in range(1, component_num):  # 对称矩阵，上三角只取半个矩阵即可
        component_bin_label = np.zeros(len(config.AU_ROI) + 1, dtype=np.uint8)
        min_box_idx = np.array(list(zip(*np.where(label_matrix == component_label)))).flatten()[0]
        for idx in set(np.array(list(zip(*np.where(label_matrix == component_label)))).flatten()):
            component_bin_label |= binary_labels[idx]
            if cal_area(*bbox[idx]) < cal_area(*bbox[min_box_idx]):
                min_box_idx = idx
        if min_box_idx not in already_idx_set:
            merged_bbox.append(bbox[min_box_idx])
            merged_label.append(component_bin_label)
            already_idx_set.add(min_box_idx)

    for idx in set(np.array(list(zip(*np.where(label_matrix == 0)))).flatten()):
        if idx not in already_idx_set:
            merged_bbox.append(bbox[idx])
            merged_label.append(binary_labels[idx])
            already_idx_set.add(idx)
    return np.array(merged_bbox), np.array(merged_label)

def show_face_box_from_mask(face_path, mask_path_dict, out_path=None, show_each_box=True):
    bbox = []
    bbox_iou_dict = {}
    i_j_dict = {}
    face_img = read_image(face_path, dtype=np.uint8, color=True).transpose(1, 2, 0)
    AU_lst = []
    for AU, mask_path in mask_path_dict.items():
        if int(AU) in list(range(51,59)):
            continue
        mask = read_image(mask_path, dtype=np.uint8, color=False)
        connect_arr = cv2.connectedComponents(mask[0], connectivity=4, ltype=cv2.CV_32S)
        component_num = connect_arr[0]
        print("AU:{0}, component_num:{1}".format(AU, component_num-1))

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
            bbox.append(coordinates)
            AU_lst.append(AU)
    bbox = np.array(bbox)
    labels = np.array(AU_lst)
    bbox, binary_labels = merge_box(bbox, labels, 0.8) # merge big IOU bbox !
    iou = bbox_iou(bbox, bbox)
    np.fill_diagonal(iou, 0)
    color_component = lambda:  random.randint(0,255)
    for i, box in enumerate(bbox):
        AU_bin = binary_labels[i]
        each_box_labels = [str(config.AU_SQUEEZE[AU_idx]) for AU_idx in np.where(AU_bin==1)[0]]
        # each box is random color
        col = (color_component(), color_component(), color_component())
        if show_each_box:
            box_face_img = face_img.copy()
            cv2.rectangle(box_face_img, (box[1], box[0]), (box[3], box[2]), col, 1)  # draw box
            cv2.putText(box_face_img, "AU:{}".format(",".join(each_box_labels)), (int((box[1] + box[3]) / 2.0),
                                                           int((box[0] + box[2]) / 2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        col, 2, cv2.LINE_AA)  # cv2.LINE_AA抗锯齿的线
        else:
            cv2.rectangle(face_img, (box[1], box[0]), (box[3], box[2]), col, 1) # draw box
            cv2.putText(face_img, "AU:{}".format(",".join(each_box_labels)), (int((box[1]+ box[3])/2.0),
                        int((box[0] + box[2]) / 2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA) # cv2.LINE_AA抗锯齿的线
        if show_each_box:
            # print(",".join(each_box_labels))

            cv2.imshow(face_path, box_face_img)
            cv2.waitKey(0)
        # if i_j_dict[i] not in already_put_iou_text_index and bbox_iou_dict[i_j_dict[i]] > 0.7:
        #     already_put_iou_text_index.add(i_j_dict[i])
        #     cv2.putText(face_img, "iou:{0:.2f}".format(bbox_iou_dict[i_j_dict[i]]), (int((box[0]+ box[2])/2.0),
        #                 int((box[1] + box[3]) / 2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA) # cv2.LINE_AA抗锯齿的线

    if out_path is not None and not show_each_box:
        cv2.imshow(face_path, face_img)
        cv2.waitKey(0)
        # cv2.imwrite(out_path, face_img)


if __name__ == "__main__":
    AU_mask_dir = "/home/machen/dataset/BP4D/BP4D_AUmask/"
    video_dir = "/home/machen/dataset/BP4D/BP4D_crop/F013/T6/"
    out_dir = "/home/machen/dataset/BP4D/BP4D_check_data/"
    au_couple_dict = get_zip_ROI_AU()
    file_lst  = os.listdir(video_dir)
    random.shuffle(file_lst)
    for file_path in file_lst:
        abs_path = video_dir + file_path
        subject_name = abs_path.split("/")[-3]
        sequence_name = abs_path.split("/")[-2]
        frame = file_path[:file_path.rindex(".")]
        au_mask_dict = {AU: "{0}/{1}/{2}/{3}_AU_{4}.png".format(AU_mask_dir,
                                                                subject_name, sequence_name,
                                                                frame, ",".join(au_couple_dict[AU])) \
                        for AU in config.AU_ROI.keys()}  # "AU": mask_path
        out_path = out_dir + file_path
        show_face_box_from_mask(abs_path, au_mask_dict, out_path, True)
        break
