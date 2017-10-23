#!/bin/bash
TOOLS="/home2/mac/caffe_mac/build/tools/"
BACKEND_DIR="CAFFE_IN"
${TOOLS}/compute_image_mean ${BACKEND_DIR}/trn.lmdb ${BACKEND_DIR}/trn_mean.binaryproto
echo "Done."
