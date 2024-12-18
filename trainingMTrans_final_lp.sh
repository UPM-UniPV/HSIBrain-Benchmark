#!/bin/bash
#conda activate MambaEnv

export MLFLOW_TRACKING_URI="file:///home/ragusa/ModelExperiments/mlruns_final/"

PATCHSIZE=7 #3
BATCHSIZE=2048 #8192

#Las Palmas
EPOCHS=300 ##250
DATASET='LP'
DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
CHANN=128

# FOLD 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P7' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'

# FOLD 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P7' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'

# FOLD 3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P7' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'

