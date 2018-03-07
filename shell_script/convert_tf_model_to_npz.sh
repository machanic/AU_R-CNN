#!/usr/bin/env bash

for folder in `ls mobilenet_trained_model`;do if [ -d mobilenet_trained_model/$folder ];then python tensorpack/scripts/dump-model-params.py --meta ./mobilenet_trained_model/${folder}/${folder}.ckpt.meta ./mobilenet_trained_model/${folder}/${folder}.ckpt.index ./mobilenet_trained_model/${folder}/tf_${folder}.npz & fi; done
