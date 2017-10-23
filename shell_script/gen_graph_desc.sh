#!/bin/bash

#nohup python structural_rnn/script/graph_desc_file_generator.py --output /home/machen/face_expr/result/graph/DISFA/ --model /home/machen/face_expr/DISFA_out/27_fold_1_resnet101_linear_model_snapshot_620000.npz --database DISFA --device 0 --proc_num 50
foldname=3_fold_1
database=BP4D
dir=${database}_${foldname}
proc=50
iter=500000
#nohup python structural_rnn/script/graph_desc_file_generator.py --output /home/machen/face_expr/result/graph/${database}/ --model /home/machen/face_expr/${dir}/${foldname}_resnet101_linear_model_snapshot_2000000.npz --database $database --device 0 --proc_num ${proc} > /dev/null 2>&1 &
mkdir  /home/machen/face_expr/result/graph/${dir}/
nohup python structural_rnn/script/graph_desc_file_generator.py -cut --output /home/machen/face_expr/result/graph/${dir}/ --model /home/machen/face_expr/${dir}/${foldname}_resnet101_linear_model_snapshot_${iter}.npz --database $database --device 0 --proc_num ${proc} > /dev/null 2>&1 &
