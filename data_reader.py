from abc import ABCMeta, abstractmethod
from collections import defaultdict
import os
import weakref
import cv2
from config import *

from faceppapi import from_face2file
#from io_utils import gen_hd5,gen_lmdb
from log_utils import LogUtils
import shutil


#from feature_extract.MDMO import MDMOFeature
from img_toolkits.other_alignment import face_alignment_flow,align_folder_firstframe_out


class DataBuilder(object):
    '''
    factory design pattern 
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def split_trn_test(self, fold):
        pass

    @abstractmethod
    def analysis_result(self):
        pass

    @abstractmethod
    def _clip_face_save(self, path):
        pass
    
    def gen_benchmark(self):
        self.analysis_result()

class CAME2DataBuilder(DataBuilder):
    def analysis_result(self):
        pass

    def split_trn_test(self):
        pass

    def __init__(self, dir):
        self.dir = dir
        self.descriptor_path = "{0}/{1}".format(dir, "CASME2-coding.csv")
        self.raw_selected = "{0}/{1}".format(dir, "CASME2_RAW_selected")
        self.cropped = "{0}/{1}".format(dir, "Cropped")
        self.mycropped_face = "{0}/{1}".format(dir, "face")
        
                
    def get_emotion_peak_imgpath(self,subject,emotion_seq):
        peak = self.emotion_peak[subject + "/" + emotion_seq][0]
        return self.mycropped_face + os.sep + "sub" + subject + os.sep + emotion_seq + os.sep + "reg_img{}.jpg".format(peak)
        
    def read(self):
        logger.log("begin read " + self.dir)
        
        self.subject_emotion  = {} # subject+"/"+emotion_seq => emotion_label_path
        self.emotion_peak = {} # subject + "/" + emotion_seq => (emotion_peak, onset_frame, offset_frame)
        with open(self.descriptor_path) as descriptor:
            for idx, line in enumerate(descriptor):
                if idx == 0:continue #skip table head
                lines = line.strip().split(",")
                self.subject_emotion[lines[0] + "/" + lines[1]] = EMOTION_LABEL_CASME2[lines[8]]
                self.emotion_peak[lines[0] + "/" + lines[1]] = (lines[4], lines[3], lines[5])
           
        self.subject_seqimg = defaultdict(list) #subject+"/"+emotion_seq => absolute_path
        
        for root,dirs,files in os.walk(self.raw_selected):
            for file in files:
                absolute_path = root+os.sep+file
                subject = absolute_path.split(os.sep)[-3].replace("sub","")
                emotion_seq = absolute_path.split(os.sep)[-2]
                file_img = absolute_path.split(os.sep)[-1]
                self.subject_seqimg[subject+"/"+emotion_seq].append(absolute_path)
                
                face_pathout = self._clip_face_save(subject, emotion_seq , absolute_path)
                
                logger.log("crop face done:{}".format(face_pathout))

        logger.log("read CAME2 done, img_size:{0} label_size:{1}".format(len(self.subject_seqimg), len(self.subject_emotion)))
        
    def _clip_face_save(self, subject, emotion_seq, path):
        path_out = self.mycropped_face + "/sub{0}_{1}_{2}".format(subject, emotion_seq.replace("_",""), os.path.basename(path))
        
        if not os.path.exists(path_out):
            #if not os.path.exists(os.path.dirname(path_out)):
            #    os.makedirs(os.path.dirname(path_out))
            logger.log("starting create face {}".format(path))
            try:
                face_img = from_face2file(path, path_out, IMG_SIZE)
            except Exception:
                shutil.copyfile(self.cropped + os.sep + "/".join(path.split("/")[-3:-1]) + os.sep + "reg_{}".format(os.path.basename(path)), path_out)
                logger.log("failure when get face++ .copy from {0} to {1}".format(self.cropped + os.sep + "/".join(path.split("/")[-3:]),
                        path_out))
                
        return path_out   


class CKPlusDataBuilder(DataBuilder):
    def __init__(self, dir):
        self.dir = dir
        self.landmark_dir = "{0}/{1}".format(dir,"Landmarks")
        self.emotion_dir = "{0}/{1}".format(dir, "Emotion")
        self.img_dir = "{0}/{1}".format(dir, "cohn-kanade-images")
        self.hd5path = "{0}/{1}".format(dir, "ck+.hd5")
    
    def read(self):
        logger.log("begin read " + self.img_dir)
        self.subject_emotion  = {} # subject+"/"+emotion_seq => emotion_label_path
        self.subject_seqimg = defaultdict(list) # subject+"/"+emotion_seq => absolute_path
        for root,dirs,files in os.walk(self.img_dir):
            for file in files:
                absolute_path = root+os.sep+file
                
                subject = absolute_path.split(os.sep)[-3]
                emotion_seq = absolute_path.split(os.sep)[-2]
                file_img = absolute_path.split(os.sep)[-1]

                emotion_label_path = self.emotion_dir+ \
                        os.sep+subject+os.sep+emotion_seq+os.sep+ \
                        file_img[:file_img.rindex(".")]+"_emotion.txt"
                if os.path.exists(emotion_label_path) and os.path.isfile(emotion_label_path):
                    self.subject_emotion[subject+"/"+emotion_seq] = emotion_label_path
                self.subject_seqimg[subject+"/"+emotion_seq].append(absolute_path)
                logger.log("read ck+ done, img_size:{0} label_size:{1}".format(len(self.subject_seqimg), len(self.subject_emotion)))
    
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
    

    @DeprecationWarning
    def split_trn_test(self, fold):
        
        count = 0
        trn = {}
        test = {}
        for key, seq_ls in self.subject_seqimg.items():
            for val in seq_ls[:6]:
                if key in self.subject_emotion:
                    count+=1
        
        for key, seq_ls in self.subject_seqimg.items():
            #abs_path is absolute_path in subject_img
            for idx, abs_path in enumerate(sorted(seq_ls, reverse=True)[:6]):
                #not all subject's emotion sequence has label
                if key not in self.subject_emotion:
                    continue
                #generate hd5 based on face region picture
                face_img_path = self._clip_face_save(abs_path)
                logger.log("clip face image path is :{}".format(face_img_path))
                if len(trn) > count * (fold - 1)/float(fold):
                    test[key+"_"+str(idx)] = (face_img_path, self._read_content(self.subject_emotion[key]))
                else:
                    trn[key+"_"+str(idx)] = (face_img_path, self._read_content(self.subject_emotion[key]))
        
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
        
    def analysis_result(self):
        pass
    
    
                
        
class DataFactory:
    _cache = weakref.WeakValueDictionary()
    
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Can't instantiate directly, pls use classmethod")
        
    @classmethod
    def get_data_builder(cls, builder_name):
        if builder_name in cls._cache:
            return cls._cache[builder_name]
        
        create_exec = READER_CREATER[builder_name]+"('"+DATA_PATH[builder_name]+"')"

        ret_obj = eval(create_exec)
        cls._cache[builder_name] = ret_obj
        return ret_obj


if __name__ == "__main__":
    face_path_ls = ["/home2/mac/testcase/face/{}".format(f) for f in sorted(os.listdir("/home2/mac/testcase/face"))]
    flow_mat_ls = ["/home2/mac/testcase/flow/{}".format(f) for f in sorted(os.listdir("/home2/mac/testcase/flow"))]
    #for path in os.listdir("/home2/mac/testcase/orig/"):
    #    from_face2file("/home2/mac/testcase/orig/"+path, "/home2/mac/testcase/face/"+path, IMG_SIZE)
    #from_face2file("D:/work/face_expression/data/CK+/cohn-kanade-images/S506/004/S506_004_00000001.png", "D:/testcase/human.jpg", IMG_SIZE)
    #test don't pass!!!
    newimg = face_alignment_flow(cv2.imread("/home2/mac/testcase/face/1.png"), cv2.imread("/home2/mac/testcase/face/S506_004_00000017.png"), "/home2/mac/testcase/flow/S506_004_00000017.png.npy")
    cv2.imshow("new", newimg)
    cv2.waitKey(0)
    #feature_extract = MDMOFeature(face_path_ls, flow_mat_ls)
    #feature = feature_extract.extract()
    #print(feature)
    #cv2.imwrite("newface.jpg",new_face)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #builder = DataFactory.get_data_builder("CAME2")
    #builder.read()
    #builder.split_trn_test(TRN_TEST_FOLD)
