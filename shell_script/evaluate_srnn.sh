#!/bin/bash
PYTHON=/usr/local/anaconda3/bin/python
database=$1
fold=3
gpu=$3
split_idx=$2
root_dir=/home3/machen/face_expr/result/graph/${database}_${fold}_fold_${split_idx}

for file in `ls $root_dir`;do
	if [[ -d ${root_dir}/${file} ]] && [[ $file == *srnn_lstm ]];then
		train_edge=all
		if [ ! "$train_edge"x = "" ];then
			train_edge=all
		fi
		if [ "$database" = "BP4D" ];then
			nohup "nohup $PYTHON structural_rnn/evaluator_main.py --target_dir ${root_dir}/${file} --bi_lstm --database $database --gpu $gpu --test ${root_dir}/test --train_edge all > ${root_dir}/eval.log 2>&1"
		else
			echo "NONE"
                        #nohup $PYTHON structural_rnn/evaluator_main.py --target_dir ${root_dir}/${file} --database $database --gpu $gpu --test ${root_dir}/test --train_edge all > ${root_dir}/eval.log 2>&1
		fi
fi
done
#model=/home/machen/face_expr/result/resnet/BP4D_3_fold_1/3_fold_1_resnet101_linear_model_snapshot_350000.npz
#out=/home/machen/face_expr/result/resnet/BP4D_3_fold_1/
#gpu=1
#nohup python -u /home/machen/face_expr/AU_rcnn/train.py --database $database --pid /tmp/AU_R_CNN/BP4D/ --need_validate --proc_num 10 --fold $fold --split_idx $split_idx --mean /home/machen/dataset/BP4D/idx/mean_no_enhance.npy --feature_model resnet101 --extract_len 1000  --pretrained_model $model  --batch_size 1 -g 1 --use_sigmoid_cross_entropy --out $out --eval_mode --use_memcached >${out}/eval.log 2>&1 &
