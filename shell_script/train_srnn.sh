#!/bin/bash

fold=3
split_idx=$1

database=$2
if [ $database == "DISFA" ]; then
root_dir="/home3/machen/face_expr/result/graph/DISFA_tmp/${database}_${fold}_fold_${split_idx}_final"
else
root_dir="/home3/machen/face_expr/result/graph/${database}_${fold}_fold_${split_idx}"
fi
train_dir="${root_dir}/train"
epoch=20
valid_dir="${root_dir}/valid"
lr=0.001

parallel_num=5
trap "exec 1000>&-;exec 1000<&-;exit 0" 2
mkfifo testfifo
exec 1000<>testfifo
rm -fr testfifo
for((n=1;n<=${parallel_num};n++))
do
    echo >&1000
done
trainer_types=(srnn)
gpu=$3
oldgpu=$3
for train_folder in `ls ${train_dir}`;
do
    if [ -d "${train_dir}/${train_folder}" ]; then
        for trainer_type in ${trainer_types[@]}
        do
        read -u1000
        if [[ "${trainer_type}" == "srnn_plus" ]] || [[ "${trainer_type}" == "srnn" ]];then
            #gpu=$(( ($gpu+1) % 5 ))
            gpu=$(( $gpu+1 ))
	    if [ $gpu -gt 9 ]; then
	        gpu=$(( oldgpu+1 ))
	    fi
        fi
        echo "using gpu "$gpu
        {
                train_keyword=${train_folder}
                # we don't need valid during train
                out_dir=${root_dir}/srnn_dropout_all
                if [ ! -d "${out_dir}" ];then
                    mkdir ${out_dir}
			echo "make dir ${out_dir}"
                fi
                if [ "${trainer_type}" == "srnn" ];then
		    if [ "${database}" == "BP4D" ]; then
                    nohup /usr/local/anaconda3/bin/python structural_rnn/train_structural_rnn_plus.py --exclude --proc_num 1 --train "${train_dir}/${train_folder}" --epoch ${epoch} --database ${database} --lr ${lr} --gpu ${gpu} --out ${out_dir}  > ${out_dir}/exclude_run_${train_keyword}.log 2>&1
		    elif [ "${database}" == "DISFA" ];then
		    nohup /usr/local/anaconda3/bin/python structural_rnn/train_structural_rnn_plus.py --proc_num 1 --train "${train_dir}/${train_folder}" --epoch ${epoch} --exclude --database ${database} --lr ${lr} --gpu ${gpu} --out ${out_dir} > ${out_dir}/exclude_run_${train_keyword}.log 2>&1;
		    fi
                    echo "train ${trainer_type} ${train_keyword} done, output log saved to ${out_dir}/run_${train_keyword}.log"
                fi
                echo >&1000
        }&
        done
    fi
done
wait
exec 1000>&-
exec 1000<&-
# --train /home/machen/face_expr/result/graph/BP4D_3_fold_1/train/5_7_4_6_9_10_27_17_23 --valid  /home/machen/face_expr/result/graph/BP4D_3_fold_1/valid/5_7_4_6_9_10_27_17_23 --epoch 20 --gpu 1 --database BP4D --lr 0.01 --with_crf
