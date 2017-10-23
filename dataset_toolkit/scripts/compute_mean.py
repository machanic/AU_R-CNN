import argparse
import sys
sys.path.insert(0,"/home/machen/face_expr/")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import chainer
from scipy import misc
import chainer.cuda as cuda
import config
from multiprocessing import Pool
import os

error_file = open("/home/machen/dataset/BP4D/idx/error_img.txt", "w")

def compute(l,r,dataset,error,N):
    sum_image = 0
    for x in range(l,r):
        try:
            image,_ = dataset[x]
        except OSError:
            path, int_label = dataset._pairs[x]
            full_path = os.path.join(dataset._root, path)
            error_file.write("{}\n".format(full_path))
            continue
        im_ = image.transpose(1,2,0)
        w,h,c = im_.shape
        l = 0
        r = w-1
        t = 0
        d = h-1
        for i in range(int(w/2)):
            if im_[i,:,:].sum()<18000:
                l = i
        for i in range(w-1,int(w/2),-1):
            if im_[i,:,:].sum()<18000:
                r = i
        for i in range(int(h/2)):
            if im_[:,i,:].sum()<18000:
                t = i
        for i in range(h-1,int(h/2),-1):
            if im_[:,i,:].sum()<18000:
                d = i
        z = im_[l+1:r-1,t+1:d-1,:]
        if(z.sum() < 18000):
            error+=1
            sum_image+= np.zeros((512, 512,3))
            continue
        image = misc.imresize(z,(512,512))
        sum_image+= image/N
        # sys.stderr.write('{} / {}\r'.format(x, N))
        # sys.stderr.flush()
    error_file.flush()
    return sum_image

def compute_mean(dataset):
    print('compute mean image')
    N = len(dataset)
    error = 0
    pool = Pool(38)
    arg = [[int(i*N/38),int((i+1)*N/38),dataset,error,N] for i in range(38)]
    sum_image = pool.starmap(compute,arg)
    pool.close()
    pool.join()
    s = np.sum(sum_image,0)
    s *= (N/(N-error))
    return np.transpose(s, (2,0,1))

def main():
    import config
    dataset = chainer.datasets.LabeledImageDataset('/home/machen/dataset/BP4D/idx/mean_no_path.txt', config.ROOT_PATH)
    mean = compute_mean(dataset)
    np.save('/home/machen/dataset/BP4D/idx/mean_no_enhance.npy', mean)
    print(mean.shape)
    print(mean.sum())
    error_file.flush()
    error_file.close()

if __name__ == "__main__":
    main()
