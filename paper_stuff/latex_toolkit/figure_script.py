import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU,get_AU_couple_child
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import cv2
from img_toolkit.face_landmark import FaceLandMark
import  numpy as np
from img_toolkit.geometry_utils import sort_clockwise
from img_toolkit.face_mask_cropper import FaceMaskCropper
from collections import OrderedDict
from img_toolkit.face_region_mask import face_img_mask
import os
import random
from AU_rcnn import transforms

YELLOW = (255, 204, 153)
RED = (255,0,0)
ORANGE = (135,184,222)
MASK_COLOR = [
(199,21,133),


]
def color_bgr(color_code):
    return (color_code[-1], color_code[-2] , color_code[-3])

def generate_landmark_image(database_name, face_img_path=None, face_img=None):
    adaptive_AU_database(database_name)
    land = FaceLandMark(
        config.DLIB_LANDMARK_PRETRAIN)
    trn_img = face_img
    if face_img is None:
        trn_img = cv2.imread(face_img_path, cv2.IMREAD_COLOR)
    landmark, _, _, landmark_txt = land.landmark(image=trn_img, ret_txt=True)
    roi_polygons = land.split_ROI(landmark)
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        # if int(roi_no) == 40 or int(roi_no) == 41:
        polygon_vertex_arr[0, :] = np.round(polygon_vertex_arr[0, :])
        polygon_vertex_arr[1, :] = np.round(polygon_vertex_arr[1, :])
        polygon_vertex_arr = sort_clockwise(polygon_vertex_arr.tolist())
        cv2.polylines(trn_img, [polygon_vertex_arr], True, 	(34,34,178), thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(trn_img, str(roi_no), tuple(np.mean(polygon_vertex_arr,axis=0).astype(np.int32)),
                    font,0.6,(0,255,255),thickness=1)
    for i, x_y in landmark_txt.items():
        x, y = x_y
        cv2.putText(trn_img, str(i), (x, y), font, 0.4, (255, 255, 255), 1)
    return trn_img

MASK_CONTAIN = {
    (199,21,133)
    :[(175,238,238),(210,105,30)]
}
def generate_mask_contain_img(database_name, img_path):
    adaptive_AU_database(database_name)
    mask_color = {}
    for parent_color, child_color in MASK_CONTAIN.items():
        mask_color[color_bgr(parent_color)] = color_bgr(child_color)
    cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img_path, channel_first=False)

    AU_couple_dict = get_zip_ROI_AU()
    AU_couple_child = get_AU_couple_child(AU_couple_dict)
    land = FaceLandMark(
        config.DLIB_LANDMARK_PRETRAIN)
    landmark, _, _ = land.landmark(image=cropped_face)
    roi_polygons = land.split_ROI(landmark)
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        polygon_vertex_arr[0, :] = np.round(polygon_vertex_arr[0, :])
        polygon_vertex_arr[1, :] = np.round(polygon_vertex_arr[1, :])
        polygon_vertex_arr = sort_clockwise(polygon_vertex_arr.tolist())
        cv2.polylines(cropped_face, [polygon_vertex_arr], True, color_bgr(RED), thickness=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cropped_face, str(roi_no), tuple(np.mean(polygon_vertex_arr, axis=0).astype(np.int32)),
                    font, 0.7, (0, 255, 255), thickness=1)
    already_fill_AU = set()
    gen_face_lst = dict()
    all_child_set = set()
    for child_set in AU_couple_child.values():
        for child in child_set:
            all_child_set.add(child)
    new_face = np.zeros_like(cropped_face)
    for AU in config.AU_ROI.keys():

        AU_couple = AU_couple_dict[AU]
        if AU_couple in all_child_set:
            continue
        if AU_couple in already_fill_AU:
            continue
        already_fill_AU.add(AU_couple)
        mask = AU_mask_dict[AU]
        child_AU_set = AU_couple_child[AU_couple]
        color_parent = list(MASK_CONTAIN.keys())[0]
        color_child = MASK_CONTAIN[color_parent]
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        color_mask[mask != 0] = color_parent
        # cv2.addWeighted(color_mask,0.5,  color_mask,1-0.5,0,color_mask)
        if np.any(new_face):
            cropped_face = new_face
        cv2.addWeighted(cropped_face, 1, color_mask,0.3, 0, new_face,-1)

        for child_AU in child_AU_set:
            if child_AU in already_fill_AU:
                continue
            already_fill_AU.add(child_AU)
            mask = AU_mask_dict[child_AU[0]]
            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            color_mask[mask != 0] = random.choice(color_child)
            cv2.addWeighted(new_face, 1, color_mask, 0.5, 0, new_face, -1)

    return new_face


def generate_AUCouple_ROI_mask_image(database_name, img_path):
    adaptive_AU_database(database_name)
    global MASK_COLOR

    mask_color_lst = []
    for color in MASK_COLOR:
        mask_color_lst.append(color_bgr(color))
    cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img_path, channel_first=False)
    AU_couple_dict = get_zip_ROI_AU()

    land = FaceLandMark(
        config.DLIB_LANDMARK_PRETRAIN)
    landmark, _, _ = land.landmark(image=cropped_face)
    roi_polygons = land.split_ROI(landmark)
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        polygon_vertex_arr[0, :] = np.round(polygon_vertex_arr[0, :])
        polygon_vertex_arr[1, :] = np.round(polygon_vertex_arr[1, :])
        polygon_vertex_arr = sort_clockwise(polygon_vertex_arr.tolist())
        cv2.polylines(cropped_face, [polygon_vertex_arr], True, color_bgr(RED), thickness=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cropped_face, str(roi_no), tuple(np.mean(polygon_vertex_arr, axis=0).astype(np.int32)),
                    font, 0.7, (0, 255, 255), thickness=1)
    already_fill_AU = set()
    idx = 0
    gen_face_lst = dict()
    for AU in config.AU_ROI.keys():
        AU_couple = AU_couple_dict[AU]
        if AU_couple in already_fill_AU:
            continue
        already_fill_AU.add(AU_couple)
        mask = AU_mask_dict[AU]
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        color_mask[mask != 0] = random.choice(mask_color_lst)
        idx += 1
        new_face = cv2.add(cropped_face, color_mask)
        gen_face_lst[AU_couple] = new_face
    return gen_face_lst

def check_box_and_cropface(orig_img_path, channel_first=False):

    cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_box(orig_img_path, channel_first=channel_first, mc_manager=None)

    AU_couple = get_zip_ROI_AU()
    already_couple = set()
    # cropped_face = np.transpose(cropped_face, (2, 0, 1))
    # cropped_face, params = transforms.random_flip(
    #     cropped_face, x_random=True, return_param=True)
    # cropped_face = np.transpose(cropped_face, (1, 2, 0))
    i = 0
    for AU, box_ls in AU_mask_dict.items():
        current_AU_couple = AU_couple[AU]
        if current_AU_couple in already_couple:
            continue
        already_couple.add(current_AU_couple)


        for box in box_ls:
            box  = np.asarray([box])

            box = transforms.flip_bbox(
                box, (512,512), x_flip=False)
            x_min,y_min = box[0][1],box[0][0]
            x_max, y_max = box[0][3],box[0][2]
            print(box)
            cp_croped = cropped_face.copy()
            cv2.rectangle(cp_croped, (x_min,y_min), (x_max, y_max),(0,255,0),1)

            cv2.imwrite("/home2/mac/test1/AU_{0}_{1}.png".format(",".join(current_AU_couple), i), cp_croped)
            i+=1
    print(i)
if __name__ == "__main__":
    adaptive_AU_database("BP4D")
    # trn_img = generate_landmark_image("BP4D","D:/Structural RNN++ paper/latex/figure/girl.jpg", None)
    # cv2.imwrite("D:/Structural RNN++ paper/latex/figure/ROI_face.jpg", trn_img)
    # exit(0)
    # imgs = ["D:/1084.jpg", "D:/007.jpg"]
    check_box_and_cropface("/home2/mac/dataset//BP4D/BP4D-training//M009/T4/0754.jpg")
    # from collections import defaultdict
    #
    # couple_mask_dict = defaultdict(list)
    # au_couple_dict = get_zip_ROI_AU()
    # for img in imgs:
    #     img_id  = img[img.index("/")+1: img.rindex(".")]
    #     cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img, channel_first=False)
    #     for AU, mask in AU_mask_dict.items():
    #         couple_mask_dict[au_couple_dict[AU]] = mask  # 所以这一步会把脸上有的，没有的AU都加上
    #     all_box_num = 0
    #     for au_couple_tuple, mask in couple_mask_dict.items():
    #         connect_arr = cv2.connectedComponents(mask, connectivity=4, ltype=cv2.CV_32S)  # mask shape = 1 x H x W
    #         component_num = connect_arr[0]
    #         label_matrix = connect_arr[1]
    #         # convert mask polygon to rectangle
    #         region_box = []  # for RNN, to sort to correct order
    #         region_label = []
    #         all_box_num += (component_num - 1)
    #         for component_label in range(1, component_num):
    #             copy_face = cropped_face.copy()
    #             row_col = list(zip(*np.where(label_matrix == component_label)))
    #             row_col = np.array(row_col)
    #             y_min_index = np.argmin(row_col[:, 0])
    #             y_min = row_col[y_min_index, 0]
    #             x_min_index = np.argmin(row_col[:, 1])
    #             x_min = row_col[x_min_index, 1]
    #             y_max_index = np.argmax(row_col[:, 0])
    #             y_max = row_col[y_max_index, 0]
    #             x_max_index = np.argmax(row_col[:, 1])
    #             x_max = row_col[x_max_index, 1]
    #             # same region may be shared by different AU, we must deal with it
    #             coordinates = (y_min, x_min, y_max, x_max)
    #             cv2.rectangle(copy_face, (x_min,y_min), (x_max,y_max), (0,0,255), thickness=1)
    #             file_name = "D:/tmp/{0}_AU_{1}({2}).jpg".format(img_id, ",".join(au_couple_tuple), component_label)
    #             cv2.imwrite(file_name, copy_face)
    #     print(all_box_num)

    # 生成各种AU couple面具图
    # root_dir = "D:/Structural RNN++ paper/latex/figure/"
    # gen_face_lst = generate_AUCouple_ROI_mask_image("BP4D", "D:/Structural RNN++ paper/latex/figure/girl.jpg")
    # for AU_couple, img in gen_face_lst.items():
    #     path = root_dir + "AU_mask/AU_{}.png".format(",".join(AU_couple))
    #     print(path)
    #     cv2.imwrite(path, img)



    # girl_face_path = root_dir+"girl.jpg"
    # new_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(girl_face_path)
    # new_face = np.transpose(new_face,(1,2,0))
    # RoI_face = generate_landmark_image(database_name="BP4D", face_img=new_face)
    # cv2.imwrite(root_dir+"ROI_face.jpg", RoI_face)