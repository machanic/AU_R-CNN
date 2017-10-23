#!/bin/bash
fold=3
gpu=0
database="BP4D"
proc_num=4
optimizer="Adam"
watch_dog_pid_file="/tmp/watch_train_${database}.pid"
watch_dog_pid=`cat ${watch_dog_pid_file}`
kill $watch_dog_pid
train_pid_folder="/tmp/AU_R_CNN/${database}/"
rm -rf ${train_pid_folder}/*

for i in `seq 3`;
do
	out_folder=${database}_${fold}_fold_${i}
	mkdir $out_folder
	nohup /usr/local/anaconda3/bin/python AU_rcnn/train.py --database $database --pid $train_pid_folder --need_validate --proc_num $proc_num --mean /home/machen/dataset/BP4D/idx/mean_no_enhance.npy --feature_model resnet101 --lr 0.0005 --extract_len 1000 --optimizer $optimizer --pretrained_model resnet101 --use_memcached --memcached_host 127.0.0.1 --fold $fold --split_idx $i --epoch 40 --batch_size 1  -g $gpu --use_sigmoid_cross_entropy --remove_non_label_frame --out ${out_folder} --snap_individual --snapshot 5000  > ${out_folder}/run_${fold}_${i}.log 2>&1 &
done
sleep 2;
/usr/local/anaconda3/bin/python train_monitor/process_monitor.py --folder $train_pid_folder --interval 600 --pid $watch_dog_pid_file
