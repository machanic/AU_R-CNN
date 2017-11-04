#!/bin/bash

fold=3
split_idx=1

database=BP4D
root_dir="/home/machen/face_expr/result/graph/${database}_${fold}_fold_${split_idx}"
train_dir="${root_dir}/train"
epoch=40
valid_dir="${root_dir}/valid"
lr=0.001
crf_lr=0.1

parallel_num=20
trap "exec 1000>&-;exec 1000<&-;exit 0" 2
mkfifo testfifo
exec 1000<>testfifo
rm -fr testfifo
for((n=1;n<=${parallel_num};n++))
do
    echo >&1000
done
trainer_type="opencrf"
for train_folder in `ls ${train_dir}`;
do
    if [ -d "${train_dir}/${train_folder}" ]; then
        read -u1000
        {
        train_keyword=${train_folder}
        # we don't need valid during train
        out_dir=${root_dir}/${trainer_type}
        if [ ! -d "${out_dir}" ];then
            mkdir ${out_dir}
        fi
        nohup /usr/local/anaconda3/bin/python structural_rnn/train_opencrf.py --proc_num 1 --train "${train_dir}/${train_folder}" --epoch ${epoch} --database ${database} --lr ${crf_lr} --need_cache_graph --out ${out_dir} > ${out_dir}/run_${train_keyword}.log 2>&1
        #echo "train ${trainer_type} ${train_keyword} done, output log saved to ${out_dir}/run_${train_keyword}.log"
        echo >&1000
        }&
    fi
done
wait
exec 1000>&-
exec 1000<&-

# --train /home/machen/face_expr/result/graph/BP4D_3_fold_1/train/5_7_4_6_9_10_27_17_23 --valid  /home/machen/face_expr/result/graph/BP4D_3_fold_1/valid/5_7_4_6_9_10_27_17_23 --epoch 20 --gpu 1 --database BP4D --lr 0.01 --with_crf
