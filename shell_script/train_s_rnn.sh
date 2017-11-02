#!/bin/bash

fold=3
split_idx=1


--train /home/machen/face_expr/result/graph/BP4D_3_fold_1/train/5_7_4_6_9_10_27_17_23 --valid  /home/machen/face_expr/result/graph/BP4D_3_fold_1/valid/5_7_4_6_9_10_27_17_23 --epoch 20 --gpu 1 --database BP4D --lr 0.01 --with_crf