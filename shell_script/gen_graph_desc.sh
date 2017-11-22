#!/bin/bash

#nohup python structural_rnn/script/graph_desc_file_generator.py --output /home/machen/face_expr/result/graph/DISFA/ --model /home/machen/face_expr/DISFA_out/27_fold_1_resnet101_linear_model_snapshot_620000.npz --database DISFA --device 0 --proc_num 50

#foldname=3_fold_1
#database=DISFA
#dir=${database}_${foldname}
#output_dir=/home/machen/face_expr/result/graph/${dir}/
#proc=30
#pretrained_model_name=resnet101
#model=/home/machen/face_expr/result/resnet/${database}_${foldname}/${foldname}_resnet101_linear_model_snapshot_800000.npz
#gpu=0
#
#echo "nohup python structural_rnn/script/graph_desc_file_generator.py --single_label -premodel $pretrained_model_name --output $output_dir --model $model --database $database --device $gpu --proc_num ${proc} > ${output_dir}/gen.log 2>&1 &"

foldname=3_fold_2
database=BP4D
dir=${database}_${foldname}
output_dir=/home3/machen/face_expr/result/graph/${dir}/
proc=30
pretrained_model_name=resnet101
model=/home/machen/face_expr/result/resnet/${database}_${foldname}/${foldname}_resnet101_linear_model_snapshot_350000.npz
gpu=2
nohup python structural_rnn/script/graph_desc_file_generator.py --single_label -premodel $pretrained_model_name --output $output_dir --model $model --database $database --device $gpu --proc_num ${proc} > ${output_dir}/gen.log 2>&1 &
