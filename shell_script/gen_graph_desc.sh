#!/bin/bash

#nohup python structural_rnn/script/graph_desc_file_generator.py --output /home/machen/face_expr/result/graph/DISFA/ --model /home/machen/face_expr/DISFA_out/27_fold_1_resnet101_linear_model_snapshot_620000.npz --database DISFA --device 0 --proc_num 50
foldname=3_fold_3
database=BP4D
dir=${database}_${foldname}
output_dir=/home/machen/face_expr/result/graph/${dir}/
proc=30
pretrained_model_name=resnet101
model=/home/machen/face_expr/result/resnet/BP4D_${foldname}/${foldname}_resnet101_linear_model_snapshot_350000.npz
gpu=0

nohup python structural_rnn/script/graph_desc_file_generator.py --single_label -premodel $pretrained_model_name --output $output_dir --model $model --database $database --device $gpu --proc_num ${proc} > ${output_dir}/gen.log 2>&1 &
