# from io_utils import gen_hd5,gen_lmdb
import sys
sys.path.insert(0, '/home/machen/face_expr')
import json
import shutil
import weakref
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np

from config import *
from img_toolkit.image_tools import get_real_img
from img_toolkit.faceppapi import from_face2file
from img_toolkit.face_crop import async_crop_face
#
# from img_toolkit.optical_flow import get_flow_mat_cache
# import random
# from operator import itemgetter
# from itertools import chain
# from scipy.sparse import vstack as sparse_vstack
# #from feature_extract.MDMO import MDMOFeature
# # from img_toolkit.other_alignment import face_alignment_flow,align_folder_firstframe_out
#



rangeid_flowmat = {}


class BaseDataReader(object):
    '''
    factory design pattern 
    '''
    __metaclass__ = ABCMeta

    def __init__(self, images, labels, video_seq, video_labels):
        self.name = None
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._images = [] # only abs_path. not real image
        self._labels = []
        self._video_seq = defaultdict(list)  # subject+"/"+emotion_seq => list of (img , absolute_path)
        self._video_labels = {}
        self._num_examples = 0

        # little trick
        images = json.loads(images)
        if images is not None:
            self._images = np.array(images)
            self._num_examples = len(self._images)
        labels = json.loads(labels)
        if labels is not None:
            self._labels = np.array(labels)
        video_seq = json.loads(video_seq)
        if video_seq is not None:
            self._video_seq = video_seq
        video_labels = json.loads(video_labels)
        if video_labels is not None:
            self._video_labels = video_labels




    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    @property
    def video_labels(self):
        return self._video_labels
    @video_labels.setter
    def video_labels(self, video_labels):
        self._video_labels = video_labels

    @property
    def video_seq(self):
        return self._video_seq

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @images.setter
    def images(self, images):
        self._images = images



    @property
    def epochs(self):
        return self._epochs_completed
    
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def _clip_face_save(self, path):
        pass


    def next_batch(self, batch_size, shuffle=True,
                   is_read_img=True, is_read_OF=False, read_npy=False):
        """Return the next `batch_size` examples from this data set."""
        read_color_type = READ_COLOR_TYPE[self.name]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]  # self._images must be np.array
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:  # not enough num_example
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)  # shuffle perm. index
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # how many still wanted
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]

            file_path_array = np.concatenate((images_rest_part, images_new_part), axis=0)
            imgs = None
            if is_read_OF:
                imgs = np.array([cv2.resize(rangeid_flowmat[of_path], IMG_SIZE)
                                 for of_path in file_path_array])
                imgs = imgs.astype(np.float32)

            elif is_read_img and not read_npy:
                imgs = np.array([cv2.resize(get_real_img(path, read_color_type), IMG_SIZE) for path in file_path_array])
                imgs = imgs.astype(np.float32)
                imgs = np.multiply(imgs, 1.0 / 255.0)
            elif read_npy:
                imgs = np.array([self.get_real_flow(abs_path=path) for path in file_path_array])

            return {"imgs": imgs, "labels": np.concatenate(
                (labels_rest_part, labels_new_part), axis=0),
                    "imgs_path": file_path_array
                    }
        else:  # has enough rest example to add
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            imgs = None

            if is_read_OF:
                imgs = np.array([cv2.resize(rangeid_flowmat[of_path], IMG_SIZE)
                                 for of_path in self._images[start:end]])
                imgs = imgs.astype(np.float32)

            elif is_read_img and not read_npy:
                imgs = np.array(
                    [cv2.resize(get_real_img(path, read_color_type), IMG_SIZE) for path in self._images[start:end]])
                imgs = imgs.astype(np.float32)
                imgs = np.multiply(imgs, 1.0 / 255.0)
            elif read_npy:
                imgs = np.array([self.get_real_flow(path) for path in self._images[start:end]])
            return {"imgs": imgs, "labels": self._labels[start:end], "imgs_path":self._images[start:end]}


class CASME2DataReader(BaseDataReader):

    def __init__(self, dir=None, images=None, labels=None, video_seq=None, video_labels=None):
        BaseDataReader.__init__(self, images, labels,video_seq, video_labels)
        self.name = "CASME2"
        if dir is not None:
            self.dir = dir
        if self._images is None:
            self._images = []
            self._labels = []
            self.num_examples = 0
            self._video_seq = defaultdict(list)  # subject+"/"+emotion_seq => list of absolute_path


        self.descriptor_path = "{0}/{1}".format(dir, "CASME2-coding.csv")
        self.raw_selected = "{0}/{1}".format(dir, "CASME2_RAW_selected")
        self.cropped = "{0}/{1}".format(dir, "Cropped")
        self.mycropped_face = "{0}/{1}".format(dir, "face")


    @property
    def num_examples(self):
        return self._num_examples

    def get_emotion_peak_imgpath(self,subject,emotion_seq):
        peak = self.emotion_peak[subject + "/" + emotion_seq][0]
        return self.mycropped_face + os.sep + "sub" + subject + os.sep + emotion_seq + \
               os.sep + "reg_img{}.jpg".format(peak)


    def read(self, clip_face=False):
        logger.log("begin read " + self.dir)
        
        self.video_labels  = {} # subject+"/"+emotion_seq => emotion_label_path
        self.emotion_peak = {} # subject + "/" + emotion_seq => (emotion_peak, onset_frame, offset_frame)

        with open(self.descriptor_path, "r") as descriptor:

            for idx, line in enumerate(descriptor):
                if idx == 0:  continue # skip table head
                lines = line.strip().split(",")
                self.video_labels[lines[0] + "/" + lines[1]] = EMOTION_LABEL_CASME2[lines[8]]
                self.emotion_peak[lines[0] + "/" + lines[1]] = (lines[4], lines[3], lines[5])
           

        
        for root,dirs,files in os.walk(self.raw_selected):
            for file in files:
                absolute_path = root+os.sep+file
                subject = absolute_path.split(os.sep)[-3].replace("sub","")
                emotion_seq = absolute_path.split(os.sep)[-2]
                file_img = absolute_path.split(os.sep)[-1]
                self._video_seq[subject+"/"+emotion_seq].append(absolute_path)
                if clip_face:
                    face_pathout = self._clip_face_save(subject, emotion_seq , absolute_path)
                    logger.log("crop face done:{}".format(face_pathout))

        for subject__emotion_seq, absolute_path_ls in self._video_seq.items():
            for absolute_path in absolute_path_ls:
                self._images.append(absolute_path)
                self._labels.append(self.video_labels[subject__emotion_seq])

        # must write! omit can cause bug!
        self._images = np.array(self._images)
        self._labels = np.array(self._labels)
        self._num_examples = self._images.size
        logger.log("read CAME2 done, img_size:{0} label_size:{1}".format(len(self._images), len(self._labels)))
        
    def _clip_face_save(self, subject, emotion_seq, path):
        path_out = self.mycropped_face + "/sub{0}_{1}_{2}".format(subject,
                                                                  emotion_seq.replace("_",""),
                                                                  os.path.basename(path))
        
        if not os.path.exists(path_out):
            # if not os.path.exists(os.path.dirname(path_out)):
            #    os.makedirs(os.path.dirname(path_out))
            logger.log("starting create face {}".format(path))
            try:
                face_img = from_face2file(path, path_out, IMG_SIZE)
            except Exception:
                shutil.copyfile(self.cropped + os.sep + "/".join(path.split("/")[-3:-1]) +
                                os.sep + "reg_{}".format(os.path.basename(path)), path_out)
                logger.log("failure when get face++ .copy from {0} to {1}".format(self.cropped
                                                                        + os.sep + "/".join(path.split("/")[-3:]),
                        path_out))
                
        return path_out


class BP4DDataReader(BaseDataReader):

    def __init__(self, dir=None, images=None, labels=None, video_seq=None, video_labels=None):

        BaseDataReader.__init__(self, images, labels, video_seq, video_labels)
        self.name = "BP4D"
        self.dir = dir
        self.img_dir = self.dir + "/BP4D_crop/"
        self.AU_occur_dir = self.dir + "/AUCoding/"
        self.AU_intensity_dir = self.dir + "AU-Intensity-Codes3.0"
        self._video_seq_dict = defaultdict(dict)  # subject+"/"+emotion_seq => {frame_no : absolute_path_list}
        self._video_seq = defaultdict(list)
        self.AU_occur_label = {} # subject + "/" + emotion_seq + "/" + frame => np.array(100)
        self.AU_intensity_label = {} # subject + "/" + emotion_seq + "/" + frame => np.array(100)

    def generate_caffe(self):

        file_writer = open("/home/machen/download/caffe-multilabel/examples/multi-label-train/input_au.txt", "w")
        for key, au_array in self.AU_occur_label.items():
            subject_name, sequence_name, frame = key.split("/")
            img_path = self._video_seq_dict[subject_name + "/" + sequence_name][int(frame)]
            au_array_str = " ".join(map(str, [e for e in au_array]))
            file_writer.write("{0} {1}\n".format(img_path, au_array_str))
            yield img_path, cv2.imread(img_path, cv2.IMREAD_COLOR), au_array
        file_writer.flush()
        file_writer.close()

    def read(self):
        logger.log("begin read AU_occurrence_dir file:{}".format(self.AU_occur_dir))
        for file_name in os.listdir(self.AU_occur_dir):
            subject_name = file_name.split("_")[0]
            sequence_name = file_name[file_name.rindex("_")+1:file_name.rindex(".")]
            video_dir = self.img_dir + os.sep + subject_name + os.sep + sequence_name + os.sep
            for img_file in sorted(os.listdir(video_dir)):
                frame = int(img_file[:img_file.rindex(".")])
                self._video_seq_dict[subject_name + "/" + sequence_name][frame] = video_dir + img_file
                self.video_seq[subject_name + "/" + sequence_name].append(video_dir + img_file)


            AU_column_idx = {}
            with open(self.AU_occur_dir+os.sep+file_name, "r") as au_file_obj:
                for idx, line in enumerate(au_file_obj):
                    if idx == 0: # header specify Action Unit
                        for col_idx, AU in enumerate(line.split(",")[1:]):
                            AU_column_idx[AU] = col_idx+1
                        continue

                    lines = line.split(",")
                    frame = lines[0]
                    # note that AU number from 1 start not 0, but numpy array index start from 0
                    # acctually in source code , did not use this au_label_array
                    au_label_array = np.array([lines[AU_column_idx[AU]] for AU in sorted(AU_ROI.keys())]).astype(np.int32)
                    self.AU_occur_label[subject_name+"/"+sequence_name+ "/" + frame] = au_label_array
                    if int(frame) not in self._video_seq_dict[subject_name+ "/"+sequence_name]:
                        print("frame not fonund!", subject_name + "/" + sequence_name + "/" + frame)
                        continue
                    self._images.append(self._video_seq_dict[subject_name + "/" + sequence_name][int(frame)])
                    self._labels.append(au_label_array)

        self._images = np.array(self._images)
        self._labels = np.array(self._labels)
        self._num_examples = self._images.size






class CKPlusDataReader(BaseDataReader):

    def __init__(self, dir=None, images=None, labels=None, video_seq=None, video_labels=None):
        self.dir = dir
        self.landmark_dir = "{0}/{1}".format(dir,"Landmarks")
        self.emotion_dir = "{0}/{1}".format(dir, "Emotion")
        self.img_dir = "{0}/{1}".format(dir, "cohn-kanade-images")
        self.hd5path = "{0}/{1}".format(dir, "ck+.hd5")
        self.AU_dir = "{0}/{1}".format(dir, "FACS")


    def get_all_au(self):
        all_AU_set = set()
        for subject_emotion, AU_path in self.subject_seq_AU.items():
            with open(AU_path, "r") as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if line != "":
                        all_AU_set.add(float(line.split()[0]))
                        if float(line.split()[0]) == 1.5: #?
                            print(AU_path)
                    else:
                        print("empty line in {}".format(AU_path))

        return all_AU_set

    def get_imgpath_AU_dict(self, last_img_num):
        imgpath_au = {}
        for key, au_path in self.subject_seq_AU.items():
            last_files = sorted(self._video_seq[key],reverse=True)[:last_img_num]
            for last_file in last_files:
                imgpath_au[last_file] = au_path
        return imgpath_au


    def get_real_au(self, path):
        au_dict = {} # AU => intensity
        with open(path, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line: continue
                AU, intensity = line.split()
                au_dict[AU] = intensity
        return au_dict

    def read(self):
        logger.log("begin read " + self.img_dir)
        self.subject_emotion  = {} # subject+"/"+emotion_seq => emotion_label_path
        self._video_seq = defaultdict(list) # subject+"/"+emotion_seq => absolute_path_list
        self.subject_seq_AU = {} #  subject+"/"+emotion_seq => AU_label_path


        for root,dirs,files in os.walk(self.img_dir):

            for file in files:
                absolute_path = root+os.sep+file
                
                subject = absolute_path.split(os.sep)[-3]
                emotion_seq = absolute_path.split(os.sep)[-2]
                file_img = absolute_path.split(os.sep)[-1]

                emotion_label_path = self.emotion_dir+ \
                        os.sep+subject+os.sep+emotion_seq+os.sep+ \
                        file_img[:file_img.rindex(".")]+"_emotion.txt"
                action_unit_dirpath = self.AU_dir + os.sep + subject + os.sep + emotion_seq + os.sep
                action_unit_filenames = os.listdir(action_unit_dirpath)
                if len(action_unit_filenames) > 0:
                    action_unit_filename = action_unit_filenames[0]
                    action_unit_path = action_unit_dirpath + action_unit_filename
                    self.subject_seq_AU[subject+"/"+emotion_seq] = action_unit_path

                if os.path.exists(emotion_label_path) and os.path.isfile(emotion_label_path):
                    self.subject_emotion[subject+"/"+emotion_seq] = emotion_label_path
                self._video_seq[subject+"/"+emotion_seq].append(absolute_path)
                logger.log("read ck+ done, img_size:{0} label_size:{1}".format(len(self._video_seq),
                                                                               len(self.subject_emotion)))
    
    def _clip_face_save(self, path):
        path_out = self.dir + os.sep + "face/" + os.path.basename(path)
        if not os.path.exists(path_out):
            face_img = from_face2file(path, path_out, IMG_SIZE)
        return path_out

    
    def _read_content(self,file_path, Dtype=int):
        content = None
        with open(file_path, "r") as file_obj:
            content = Dtype(float(file_obj.read().strip()))
        return content
    

        '''
        if BACKEND == "lmdb":
            gen_lmdb("trn", trn)
            gen_lmdb("test", test)
            logger.log("all file lmdb generate done")
        elif BACKEND == "hdf5":
            gen_hd5("trn", trn) #generate hd5 based on face region picture
            gen_hd5("test", test)
            logger.log("all file hd5 generate done")
        '''
  
        return {"trn":trn, "test":test}


def is_all_list(x):
    return isinstance(x, list)

def flatten(sequence, to_expand=is_all_list,want_idx=None):
    for item in sequence:
        if to_expand(item):
            for subitem in flatten(item, to_expand):
                if want_idx is not None and isinstance(subitem[want_idx], list):
                    for subsubitem in subitem[want_idx]:
                        yield subsubitem
                elif want_idx is not None and isinstance(subitem[want_idx],str):
                        yield subitem[want_idx]
                else:
                    yield subitem
        else:
            if want_idx is not None and isinstance(item[want_idx], list):
                for subsubitem in item[want_idx]:
                    yield subsubitem
            elif want_idx is not None and isinstance(item[want_idx], str):
                yield item[want_idx]
            else:
                yield item
        
class DataFactory:
    _cache = weakref.WeakValueDictionary()

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Can't instantiate directly, pls use classmethod")
        
    @classmethod
    def get_data_reader(cls, builder_name, imgs=None, labels=None,video_seq=None,video_labels=None):
        # if imgs is not None, it must recreate data_reader
        if builder_name in cls._cache:
            return cls._cache[builder_name]
        if isinstance(imgs, np.ndarray):
            imgs = imgs.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        end_index = len(builder_name) if '-' not in builder_name else builder_name.index('-')

        create_exec = READER_CREATER[builder_name[0:end_index]] + \
                        "('" + DATA_PATH[builder_name[0:end_index]]+"','" + \
                        json.dumps(imgs) + "','" + \
                        json.dumps(labels) + "','"+ \
                        json.dumps(video_seq) +"','" + json.dumps(video_labels)+"')"
        create_exec = create_exec.replace("\\", "/")
        ret_obj = eval(create_exec)

        cls._cache[builder_name] = ret_obj
        return ret_obj


    @classmethod
    def kfold_video_split(cls, builder_name, fold, shuffle=True):
        try:
            data_reader = cls._cache[builder_name]
            names = ['keyid', 'video_seq']
            formats = ['S32', 'list']
            dtype = dict(names=names, formats=formats)

            video_seq_arr = list(data_reader.video_seq.items())
        except KeyError:
            raise KeyError("must build basic data builder:{} first!".format(builder_name))
        if shuffle:
            perm = np.arange(len(video_seq_arr))
            np.random.shuffle(perm)
            new_video_seq_arr = []
            for each_perm in perm:
                new_video_seq_arr.append(video_seq_arr[each_perm])
            video_seq_arr = new_video_seq_arr
        per_fold = len(video_seq_arr) // fold


        kfold_imgs = [video_seq_arr[i * per_fold:(i + 1) * per_fold]
                               for i in range(fold)]
        orig_labels = []
        for i in range(fold):
            for j in range(i * per_fold, (i+1) * per_fold):
                extend_len = len(video_seq_arr[j][1])
                subject_seq = video_seq_arr[j][0]
                video_fold_labels = [data_reader.video_labels[subject_seq]] * extend_len
                orig_labels.append(video_fold_labels)
        kfold_labels = [orig_labels[i * per_fold: (i+1) * per_fold] for i in range(fold)]

        kfold_samples = []
        for fold_idx, fold_imgs in enumerate(kfold_imgs):
            kfold_samples.append({"trn": cls.get_data_reader("{0}-trn-video_fold:{1}".format(builder_name, fold_idx),
                                                             list(flatten(kfold_imgs[0:fold_idx] + kfold_imgs[fold_idx+1:],want_idx=1)),
                                                             list(flatten(kfold_labels[0:fold_idx] + kfold_labels[fold_idx+1:])),
                                                             {k:data_reader.video_seq[k] for k in set(flatten(
                                                                 kfold_imgs[0:fold_idx] + kfold_imgs[fold_idx + 1:],
                                                                 want_idx=0))},
                                                             None),
                                                             # {k: data_reader.video_labels[k] for k in set(flatten(
                                                             #     kfold_imgs[0:fold_idx] + kfold_imgs[fold_idx + 1:],
                                                             #     want_idx=0))}),
                                  "test": cls.get_data_reader("{0}-test-video_fold:{1}".format(builder_name, fold_idx),
                                                              list(flatten(fold_imgs, want_idx=1)),
                                                              list(flatten(kfold_labels[fold_idx])),
                                                              {k: data_reader.video_seq[k] for k in set(flatten(
                                                                  kfold_imgs[fold_idx],
                                                                  want_idx=0))}, None
                                                              )})
                                                              # {k: data_reader.video_labels[k] for k in set(flatten(
                                                              #     kfold_imgs[fold_idx],
                                                              #     want_idx=0))})})
        return kfold_samples


    #MUST BE CALLED AFTER  def get_data_builder
    @classmethod
    def kfold_split(cls, builder_name,  fold, shuffle=True):
        try:
            data_builder = cls._cache[builder_name]
        except KeyError:
            raise KeyError("must build basic data builder:{} first!".format(builder_name))
        if shuffle:
            perm = np.arange(data_builder._num_examples)
            np.random.shuffle(perm)
            data_builder._images = data_builder.images[perm]
            data_builder._labels = data_builder.labels[perm]
        '''
        >>> np.array([[1,2,3],[3,4]])
        array([[1, 2, 3], [3, 4]], dtype=object)
        >>> a = np.array([[1,2,3],[3,4]])
        >>> a.flatten()  #don't work!
        array([[1, 2, 3], [3, 4]], dtype=object)
        '''
        per_fold = data_builder._num_examples // fold
        kfold_imgs = np.array([data_builder._images[i*per_fold:(i+1)*per_fold] for i in range(fold)])
        kfold_labels = np.array([data_builder._labels[i*per_fold:(i+1)*per_fold] for i in range(fold)])
        kfold_samples = []
        for fold_idx, fold_imgs in enumerate(kfold_imgs):

            kfold_samples.append({"trn": cls.get_data_reader("{0}-trn-image_fold:{1}".format(builder_name, fold_idx),
                                                             np.concatenate([kfold_imgs[0:fold_idx], kfold_imgs[fold_idx+1:]]).flatten(),
                                                             np.concatenate([kfold_labels[0:fold_idx], kfold_labels[fold_idx + 1:]]).flatten()
                                                             ),
                                  "test": cls.get_data_reader("{0}-test-image_fold:{1}".format(builder_name, fold_idx),
                                                              fold_imgs, kfold_labels[fold_idx]
                                                              )})
        return kfold_samples


if __name__ == "__main__":

    async_crop_face("BP4D", DATA_PATH["BP4D"] + os.sep + "/release/BP4D-training/", 100, force_write=True)


    # print(len(builder.images))
    # d = DataFactory.kfold_split("CASME2",5,False)
    # for f in d:
    #     trn = f["trn"]
    #     test = f["test"]
    #     strn = set(trn.images)
    #     stest = set(test.images)
    #     print(strn & stest)
    #     print(len(trn.images))
    #     print("________")
    #     print(len(test.images))
    #     break
    #     # while trn.epochs < 2:
        #     trn.next_batch(30, is_read_file=False)
        # n = trn.next_batch(30, is_read_file=False)
        # print(n["imgs_path"], n["labels"])
        # print(len(trn.images))

    #face_path_ls = ["/home2/mac/testcase/face/{}".format(f) for f in sorted(os.listdir("/home2/mac/testcase/face"))]
    #flow_mat_ls = ["/home2/mac/testcase/flow/{}".format(f) for f in sorted(os.listdir("/home2/mac/testcase/flow"))]
    #for path in os.listdir("/home2/mac/testcase/orig/"):
    #    from_face2file("/home2/mac/testcase/orig/"+path, "/home2/mac/testcase/face/"+path, IMG_SIZE)
    #from_face2file("D:/work/face_expression/data/CK+/cohn-kanade-images/S506/004/S506_004_00000001.png", "D:/testcase/human.jpg", IMG_SIZE)
    #test don't pass!!!
    # newimg = face_alignment_flow(cv2.imread("/home2/mac/testcase/face/1.png"), cv2.imread("/home2/mac/testcase/face/S506_004_00000017.png"), "/home2/mac/testcase/flow/S506_004_00000017.png.npy")
    # cv2.imshow("new", newimg)
    # cv2.waitKey(0)
    #feature_extract = MDMOFeature(face_path_ls, flow_mat_ls)
    #feature = feature_extract.extract()
    #print(feature)
    #cv2.imwrite("newface.jpg",new_face)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #builder = DataFactory.get_data_builder("CAME2")
    #builder.read()
    #builder.split_trn_test(TRN_TEST_FOLD)
