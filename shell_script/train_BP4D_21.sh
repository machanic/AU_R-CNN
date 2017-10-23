#!/bin/bash
#for i in `seq 10`
#do
#	nohup /usr/local/anaconda3/bin/python AU_rcnn/train.py --feature_model resnet101 --extract_len 1000 --optimizer RMSprop --pretrained_model resnet101 --use_memcached --memcached_host 127.0.0.1 --fold 10 --split_idx $i --epoch 20 --batch_size 4 --step_size 5000 -g0,1 >result/run_10_$i.log 2>&1 
#done

nohup /usr/local/anaconda3/bin/python AU_rcnn/train.py --feature_model resnet101 --lr 0.005 -proc 3 --extract_len 1000 --optimizer RMSprop --pretrained_model resnet101 --use_memcached --memcached_host 127.0.0.1 --fold 21 --split_idx 21 --epoch 50 --batch_size 1  --step_size 5000 -g 0 >result/run_21_21.log 2>&1 &
