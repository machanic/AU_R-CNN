import glob
import multiprocessing as mp
import os
import traceback

import cv2
import numpy as np

# from config import IMG_SIZE, DLIB_LANDMARK_PRETRAIN, AU_ROI,AU_REGION_MASK_PATH,CROP_DATA_PATH, DATA_PATH
import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU
from img_toolkit.face_landmark import FaceLandMark
from img_toolkit.face_region_mask import crop_face_img_mask


# def face_alignment_flow(img1, img2, flow_path):
#     flow_result = np.load(flow_path)
#     landmark, new_face = MDMOFeature.land.landmark(image=img1)
#
#     # print img1.shapes
#     # print(landmark)
#     contour_landmark = [i for i in range(1, 16)] + [30]
#     # feature_pos each is (x,y)
#     feature_pos = [list(landmark[no]) for no in contour_landmark]
#     # print(feature_pos)
#     feature_back = []
#     for pos in feature_pos:
#         pos[0] = max(pos[0], 0)
#         pos[1] = max(pos[1], 0)
#         pos[0] = min(pos[0], img2.shape[1] - 1)
#         pos[1] = min(pos[1], img2.shape[0] - 1)
#         new_pos = (pos[0] + flow_result[pos[1], pos[0]][0],
#                    pos[1] + flow_result[pos[1], pos[0]][1])
#         feature_back.append(new_pos)
#     convert_mat = fl.calc_affine_mat(
#         np.array(
#             feature_back, dtype=np.float32), np.array(
#             feature_pos, dtype=np.float32))
#     # print(convert_mat)
#     B = cv2.warpPerspective(
#         img2, convert_mat.T, (img2.shape[1], img2.shape[0]))
#     return B

def sub_process_miss_mask(database, landmark, orig_file_path_lst, AU_couple_dict):

    for orig_file_path in orig_file_path_lst:
        print("processing {}".format(orig_file_path))
        subject_name = orig_file_path.split("/")[-3]
        sequenece_name = orig_file_path.split("/")[-2]
        frame = orig_file_path.split("/")[-1]
        orig_img = cv2.imread(orig_file_path, cv2.IMREAD_COLOR)

        new_face, rect = dlib_face_rect(orig_img, landmark)
        new_face = cv2.resize(new_face, config.IMG_SIZE)  # note that mask png also needs to resize

        crop_file_path = config.CROP_DATA_PATH[database] + os.sep + subject_name + os.sep + sequenece_name + os.sep + frame
        cv2.imwrite(crop_file_path, new_face)
        print("dlib crop {} done".format(crop_file_path))
        already_mask = set()
        for AU, au_couple in AU_couple_dict.items():
            if au_couple in already_mask:
                continue
            already_mask.add(au_couple)
            au_mask_dir = "{0}/{1}/{2}/".format(config.AU_REGION_MASK_PATH[database],
                                                subject_name, sequenece_name)
            au_couple = ",".join(au_couple)
            filename = os.path.basename(crop_file_path)
            filename = filename[:filename.rindex(".")]

            au_mask_path = "{0}/{1}_AU_{2}.png".format(au_mask_dir, filename, au_couple)
            # because of resize, rect also needs to modify coordinates

            mask = crop_face_img_mask(AU, orig_img, new_face, rect, landmarker=landmark)
            cv2.imwrite(au_mask_path, mask)
            print("write {}".format(au_mask_path))

def regenerate_miss_mask(database, mp_num, miss_file_path):
    pool = mp.Pool()
    AU_couple_dict = get_zip_ROI_AU()
    orig_file_path_lst = set()
    landmark = FaceLandMark(config.DLIB_LANDMARK_PRETRAIN)
    with open(miss_file_path, 'r') as file_obj:
        for line in file_obj:
            miss_mask_path = line.strip()
            subject_name = miss_mask_path.split("/")[-3]
            sequenece_name = miss_mask_path.split("/")[-2]
            frame = miss_mask_path.split("/")[-1]
            frame = frame[:frame.index("_")]
            orig_train_file_path = config.DATA_PATH["BP4D"] + "/release/BP4D-training/{0}/{1}/{2}.jpg".format(subject_name,
                                                                                                       sequenece_name, frame)
            orig_file_path_lst.add(orig_train_file_path)

    split_list = lambda A, n=3: [A[i:i + n] for i in range(0, len(A), n)]
    sub_list = split_list(list(orig_file_path_lst), len(orig_file_path_lst) // mp_num)
    for sub in sub_list:
        pool.apply_async(sub_process_miss_mask, args=(database, landmark, sub, AU_couple_dict))
    pool.close()
    pool.join()










def dlib_face_rect(image, landmark):
    h_offset = 50
    w_offset = 20
    try:
        landmark_dict, origimg, _ = landmark.landmark(image=image)

    except IndexError:
        traceback.print_exc()
        return image, {"top":0, "left":0, "height": image.shape[0], "width": image.shape[1]}
    sorted_x = sorted([val[0] for val in landmark_dict.values()])
    sorted_y = sorted([val[1] for val in landmark_dict.values()])
    rect = {"top": sorted_y[0]-h_offset, "left":sorted_x[0]-w_offset,
            "width": sorted_x[-1] - sorted_x[0]+2*w_offset, "height": sorted_y[-1] - sorted_y[0]+h_offset}
    for key, val in rect.items():
        if val < 0:
            rect[key] = 0
    new_face = image[rect["top"]:rect["top"]+rect["height"], rect["left"]: rect["left"] + rect["width"], ...]
    return new_face, rect

def face_alignment_landmark(img,  src_landmark_points, dst_landmarks_points):
    affine_mat = cv2.getAffineTransform(np.array(src_landmark_points), np.array(dst_landmarks_points))
    return cv2.warpAffine(img, affine_mat, (img.shape[0], img.shape[1]))


# def align_folder_firstframe(img_folder_path_ls, flow_foler_path_ls):
#     newimg_ls = []
#     img1_path = img_folder_path_ls[0]
#     for idx, img_path in enumerate(
#             img_folder_path_ls[1:]):  # idx start from 1, which indicate 2nd picture
#         img1 = cv2.imread(img1_path)
#         img2 = cv2.imread(img_path)
#         newimg = face_alignment_flow(img1, img2, flow_foler_path_ls[idx])
#         newimg_ls.append(newimg)
#     return newimg_ls


def async_crop_face(database, src_img_dir, mp_num, force_write=True):
    pool = mp.Pool()
    data_path_lst = set()
    not_contain_count = 0
    contain_count = 0
    AU_couple_dict = get_zip_ROI_AU()
    for root, dirs, files in os.walk(src_img_dir):
        for file in files:
            absolute_path = root + os.sep + file
            subject_name = absolute_path.split(os.sep)[-3]
            sequence = absolute_path.split(os.sep)[-2]

            dst_dir = config.CROP_DATA_PATH[database] + os.sep + subject_name + os.sep + sequence + os.sep
            cropped_path = dst_dir + os.sep + os.path.basename(absolute_path)
            data_path_lst.add(absolute_path)

    split_list = lambda A, n=3: [A[i:i + n] for i in range(0, len(A), n)]
    print(len(data_path_lst)//mp_num)
    sub_list = split_list(list(data_path_lst), len(data_path_lst)//mp_num)
    for sub in sub_list:
        pool.apply_async(sub_process, args=(database, sub, force_write))
    pool.close()
    pool.join()


def sub_process(database, file_list, force_write):
    for _ in generate_face_AUregion_mask(database, generate_crop_recurive(database, file_list, force_write)):
        pass  # because of yield



def generate_crop_recurive(database, iter_filepath, force_write):

    landmark = FaceLandMark(config.DLIB_LANDMARK_PRETRAIN)
    for absolute_path in iter_filepath:

        try:
            if absolute_path.endswith(".jpg") or absolute_path.endswith(".png"):
                file = os.path.basename(absolute_path)

                subject_name = absolute_path.split(os.sep)[-3]
                sequence = absolute_path.split(os.sep)[-2]
                dst_dir = config.CROP_DATA_PATH[database] + os.sep + subject_name + os.sep + sequence + os.sep

                # if not force_write and os.path.exists(dst_dir+os.sep+file):
                #     continue
                if force_write or not os.path.exists(dst_dir + os.sep + file):
                    orig_img = cv2.imread(absolute_path, cv2.IMREAD_COLOR)
                    new_face, rect = dlib_face_rect(orig_img, landmark) # call dlib first detect feature point
                    new_face = cv2.resize(new_face, config.IMG_SIZE) # then resize

                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    print("write : {}".format(dst_dir + os.sep + file))
                    # cv2.imwrite(dst_dir+os.sep+file, new_face)  # 临时删掉了，记得恢复这句话
                    yield new_face, orig_img, rect, dst_dir+os.sep+file
        except IndexError:
            continue


# use togather with folder_crop_recurive, parameter is a list generator
def generate_face_AUregion_mask(database, iter_crop):

    landmark = FaceLandMark(config.DLIB_LANDMARK_PRETRAIN)
    AU_couple_dict = get_zip_ROI_AU()
    already_mask = set()
    for new_face, orig_img, rect, absolute_path in iter_crop:  # absolute_path 是截取crop新脸的路径
        try:
            absolute_path = absolute_path.replace("//", "/")

            subject_name = absolute_path.split(os.sep)[-3]
            sequence = absolute_path.split(os.sep)[-2]
            filename = os.path.basename(absolute_path)
            filename = filename[:filename.rindex(".")]
            au_mask_dir = "{0}/{1}/{2}/".format(config.AU_REGION_MASK_PATH[database],
                                                subject_name, sequence)
            for f in glob.glob("{0}/{1}_AU_*".format(au_mask_dir, filename)):
                os.remove(f)
            for AU in config.AU_ROI.keys():
                if AU_couple_dict[AU] in already_mask:
                    continue
                mask = crop_face_img_mask(AU, orig_img, new_face, rect, landmarker=landmark)
                if not os.path.exists(au_mask_dir):
                    os.makedirs(au_mask_dir)
                    print("make dir {}".format(au_mask_dir))
                au_couple = AU_couple_dict[AU]
                already_mask.add(au_couple)
                au_couple = ",".join(au_couple)
                au_mask_path = "{0}/{1}_AU_{2}.png".format(au_mask_dir, filename, au_couple)

                cv2.imwrite(au_mask_path, mask)
                print("write : {}".format(au_mask_path))
                yield mask, au_mask_path
            already_mask.clear()
        except IndexError:
            continue

if __name__ == "__main__":
    regenerate_miss_mask("BP4D", 10,  "/tmp/error_mask.log")
#
# def align_folder_firstframe_out(
#         img_folder_path_ls,
#         flow_folder_path_ls,
#         out_folder_path):
#     newimg_ls = align_folder_firstframe(
#         img_folder_path_ls, flow_folder_path_ls)
#     for idx, newimg in enumerate(newimg_ls):
#         cv2.imwrite(os.sep.join(
#             (out_folder_path, os.path.basename(img_folder_path_ls[idx + 1]))), newimg)
