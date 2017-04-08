
CAFFE=/home2/mac/caffe_mac/build/tools/caffe
$CAFFE train -solver alexnet_solver.prototxt -gpu 1 -weights /home/mac/data/bvlc_alexnet.caffemodel