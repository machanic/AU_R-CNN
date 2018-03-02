#!/usr/bin/env bash
gpu=$2 # pass as argument
database=$1  # BP4D or DISFA

epoch=25
lr=0.01

### temporal_edge_mode = rnn
neighbor_mode=attention_fuse
spatial_edge_mode=all_edge
temporal_edge_mode=rnn

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


neighbor_mode=concat_all
spatial_edge_mode=all_edge
temporal_edge_mode=rnn

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=random_neighbor
spatial_edge_mode=all_edge
temporal_edge_mode=rnn
for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

# this is special case
no_neighbor=no_neighbor
spatial_edge_mode=no_edge
temporal_edge_mode=rnn
for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


neighbor_mode=attention_fuse
spatial_edge_mode=no_edge
temporal_edge_mode=rnn

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=concat_all
spatial_edge_mode=no_edge
temporal_edge_mode=rnn

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=random_neighbor
spatial_edge_mode=no_edge
temporal_edge_mode=rnn

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


### temporal_edge_mode = attention_block
neighbor_mode=attention_fuse
spatial_edge_mode=all_edge
temporal_edge_mode=attention_block

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


neighbor_mode=concat_all
spatial_edge_mode=all_edge
temporal_edge_mode=attention_block

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=random_neighbor
spatial_edge_mode=all_edge
temporal_edge_mode=attention_block
for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

# this is special case
no_neighbor=no_neighbor
spatial_edge_mode=no_edge
temporal_edge_mode=attention_block
for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


neighbor_mode=attention_fuse
spatial_edge_mode=no_edge
temporal_edge_mode=attention_block

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=concat_all
spatial_edge_mode=no_edge
temporal_edge_mode=attention_block

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=random_neighbor
spatial_edge_mode=no_edge
temporal_edge_mode=attention_block

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done



### temporal_edge_mode = no_temporal

neighbor_mode=attention_fuse
spatial_edge_mode=all_edge
temporal_edge_mode=no_temporal

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


neighbor_mode=concat_all
spatial_edge_mode=all_edge
temporal_edge_mode=no_temporal

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=random_neighbor
spatial_edge_mode=all_edge
temporal_edge_mode=no_temporal
for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

# this is special case
no_neighbor=no_neighbor
spatial_edge_mode=no_edge
temporal_edge_mode=no_temporal
for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done


neighbor_mode=attention_fuse
spatial_edge_mode=no_edge
temporal_edge_mode=no_temporal

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=concat_all
spatial_edge_mode=no_edge
temporal_edge_mode=no_temporal

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done

neighbor_mode=random_neighbor
spatial_edge_mode=no_edge
temporal_edge_mode=no_temporal

for split_idx in `seq 3`;do
    train_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/train
    test_dir=/home/machen/dataset/graph/${database}_3_fold_${split_idx}/test
	/usr/bin/python /home/machen/face_expr/graph_learning/train_st_attention_net.py -g $gpu --lr $lr --epoch $epoch --train $train_dir --test $test_dir --database $database --neighbor_mode $neighbor_mode --spatial_edge_mode $spatial_edge_mode --temporal_edge_mode $temporal_edge_mode
done
