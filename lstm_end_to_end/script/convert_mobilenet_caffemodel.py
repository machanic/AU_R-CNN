
from chainer.links.caffe.caffe_function import CaffeFunction

def load_caffemodel(file_path):
    caffemodel = CaffeFunction("mobilenet.caffemodel")

