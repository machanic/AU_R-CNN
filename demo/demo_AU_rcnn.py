import argparse
from img_toolkit.face_landmark import FaceLandMark
import cv2
import numpy as np
import os
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from img_toolkit.geometry_utils import sort_clockwise
import config
from img_toolkit.face_mask_cropper import FaceMaskCropper
from AU_rcnn.links.model.faster_rcnn import FasterRCNNResnet101
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_vgg import FasterRCNNVGG16
from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
import chainer
from chainer.dataset.convert import concat_examples
import matplotlib.pyplot as plt
import math
from collections import defaultdict

def generate_AUCouple_ROI_mask_image(database_name, img_path, roi_activate):
    adaptive_AU_database(database_name)

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
        cv2.polylines(cropped_face, [polygon_vertex_arr], True, (0,0,255), thickness=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cropped_face, str(roi_no), tuple(np.mean(polygon_vertex_arr, axis=0).astype(np.int32)),
                    font, 0.7, (0, 255, 255), thickness=1)
    already_fill_AU = set()
    AUCouple_face_dict = dict()
    for AU in config.AU_ROI.keys():
        AU_couple = AU_couple_dict[AU]
        if AU_couple in already_fill_AU or AU_couple not in roi_activate:
            continue
        already_fill_AU.add(AU_couple)
        mask = AU_mask_dict[AU]
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        color_mask[mask != 0] = (199,21,133)
        new_face = cv2.add(cropped_face, color_mask)
        AUCouple_face_dict[AU_couple] = new_face

    return AUCouple_face_dict


def main():
    parser = argparse.ArgumentParser(
        description='generate Graph desc file script')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_no_enhance.npy",
                        help='image mean .npy file')
    parser.add_argument("--image", default='C:/Users/machen/Downloads/tmp/face.jpg')
    parser.add_argument("--model",
                        default="C:/Users/machen/Downloads/tmp/BP4D_3_fold_1.npz")
    parser.add_argument("--pretrained_model_name", '-premodel', default='resnet101')
    parser.add_argument('--database', default='BP4D',
                        help='Output directory')
    parser.add_argument('--device', default=0, type=int,
                        help='GPU device number')
    args = parser.parse_args()
    adaptive_AU_database(args.database)

    if args.pretrained_model_name == "resnet101":
        faster_rcnn = FasterRCNNResnet101(n_fg_class=len(config.AU_SQUEEZE),
                                          pretrained_model="resnet101",
                                          mean_file=args.mean, use_lstm=False,
                                          extract_len=1000)  # 可改为/home/machen/face_expr/result/snapshot_model.npz
    elif args.pretrained_model_name == "vgg":
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model="imagenet",
                                      mean_file=args.mean,
                                      use_lstm=False,
                                      extract_len=1000)

    if os.path.exists(args.model):
        print("loading pretrained snapshot:{}".format(args.model))
        chainer.serializers.load_npz(args.model, faster_rcnn)
    if args.device >= 0:
        faster_rcnn.to_gpu(args.device)
        chainer.cuda.get_device_from_id(int(args.device)).use()


    cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(args.image, channel_first=True)
    au_couple_dict = get_zip_ROI_AU()
    au_couple_child = get_AU_couple_child(au_couple_dict)  # AU couple tuple => child fetch list
    au_couple_box = dict()  # value is box (4 tuple coordinate) list

    for AU, AU_couple in au_couple_dict.items():
        au_couple_box[AU_couple] = AU_box_dict[AU]
    box_lst = []
    roi_no_AU_couple_dict = dict()
    roi_no = 0
    for AU_couple, couple_box_lst in au_couple_box.items():
        box_lst.extend(couple_box_lst)
        for _ in couple_box_lst:
            roi_no_AU_couple_dict[roi_no] = AU_couple
            roi_no += 1

    box_lst = np.asarray(box_lst)
    cropped_face = cropped_face.astype(np.float32)
    cropped_face = faster_rcnn.prepare(cropped_face)  # substract mean pixel value
    box_lst = box_lst.astype(np.float32)
    batch = [(cropped_face, box_lst), ]
    cropped_face, box_lst = concat_examples(batch, args.device)

    if box_lst.shape[1] != config.BOX_NUM[args.database]:
        print("error box num {0} != {1}".format(box_lst.shape[1], config.BOX_NUM[args.database]))
        return
    with chainer.no_backprop_mode():
        roi_preds, _ = faster_rcnn.predict(cropped_face, box_lst)
    roi_preds = roi_preds.reshape(box_lst.shape[1], len(config.AU_SQUEEZE))  # shape = R, Y
    roi_preds = chainer.cuda.to_cpu(roi_preds)
    nonzero_idx = np.nonzero(roi_preds)
    activate_couple = set()
    for roi_no in nonzero_idx[0]:
        activate_couple.add(roi_no_AU_couple_dict[roi_no])
    roi_activate = list(activate_couple)


    AUCouple_face_dict = generate_AUCouple_ROI_mask_image(database_name=args.database,
                                                          img_path=args.image, roi_activate=roi_activate)  # AU couple and face
    preds = np.bitwise_or.reduce(roi_preds, axis=0)  # shape = Y
    assert preds.shape[0] == len(config.AU_SQUEEZE)
    image_pred_label = list()
    nonzero_preds = np.nonzero(preds)
    for AU_idx in nonzero_preds[0]:
        image_pred_label.append(config.AU_SQUEEZE[AU_idx])

    show_label_subimage = defaultdict(list)
    for AU_couple in AUCouple_face_dict.keys():
        for AU in AU_couple:
            if AU in image_pred_label:
                show_label_subimage[AU_couple].append(AU)

    total_num = len(show_label_subimage)
    f,ax = plt.subplots(int(math.ceil(total_num/2)),2)
    del_AU_couple = list()  # remove the effect of "label fetch", remove unrelated AU
    for AU_couple in AUCouple_face_dict.keys():
        if AU_couple not in show_label_subimage:
            del_AU_couple.append(AU_couple)
    for AU_couple in del_AU_couple:
        del AUCouple_face_dict[AU_couple]
    AUCouple_face_dict = list(AUCouple_face_dict.items())
    assert len(AUCouple_face_dict) == total_num
    for i in range(total_num):
        ax[i//2,i%2].set_title("AU:{}".format(show_label_subimage[AUCouple_face_dict[i][0]]))
        img = AUCouple_face_dict[i][1]  # OpenCV is in BGR mode
        img = img[:,:,::-1]  # convert BGR to RGB in Channel axis
        ax[i//2,i%2].imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
