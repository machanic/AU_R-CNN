#!/bin/bash
fold=27
gpu=1
database="DISFA"
for i in `seq 4`;
do
	nohup /usr/local/anaconda3/bin/python AU_rcnn/train.py --mean /home/machen/dataset/BP4D/idx/mean_no_enhance.npy --feature_model resnet101 --lr 0.001 --extract_len 1000 --optimizer RMSprop --pretrained_model resnet101 -proc 10 --use_memcached --memcached_host 127.0.0.1 --fold $fold --split_idx $i --epoch 40 --batch_size 1  --step_size 5000 -g $gpu --use_sigmoid_cross_entropy --remove_non_label_frame --out ${database}_out --database ${database} > ${database}_out/run_${fold}_${i}.log 2>&1 & 
done
