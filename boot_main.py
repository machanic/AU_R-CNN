# from feature_extract.REINFORCE_MD import WordBook, ExpressionDescritor
# from img_toolkit.image_tools import get_real_img
import numpy as np

from dataset_toolkit.data_reader import DataFactory

if __name__ == "__main__":
    reader = DataFactory.get_data_reader("ck+")
    reader.read()
    au_max = max(reader.get_all_au())
    print(reader.get_all_au())
    au = reader.get_imgpath_AU_dict(last_img_num=4)

    caffe_file_writer = open("D:/file_list.txt", "w")
    for img_path, au_path in au.items():
        au_vec = np.zeros(int(au_max))
        au_vec -= 1
        au_lst = reader.get_real_au(au_path).keys()
        for au in au_lst:
            au = int(float(au))
            np.put(au_vec, au-1, 1)
        line = "{0} {1}\n".format(img_path, " ".join(map(str, map(int,au_vec))))
        line = line.replace("\\", "/")
        caffe_file_writer.write(line)
    caffe_file_writer.flush()
    caffe_file_writer.close()

    # kfold_samples = DataFactory.kfold_video_split("CASME2", 6 , True)
    #
    # trn_data_set = kfold_samples[0]["trn"]
    # test_data_set = kfold_samples[0]["test"]
    # book = WordBook(trn_data_set)
    # ed = ExpressionDescritor(book)
    #
    # test_feature_list = []
    # for key, video_seq in test_data_set.video_seq.items():
    #     test_feature_list.append(ed.hist(key, video_seq))
    #
    # for f in test_feature_list:
    #     print(f)