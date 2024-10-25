#!/bin/bash
#conda activate MambaEnv

PATCHSIZE=3
BATCHSIZE=8192

#Madrid
EPOCHS=250
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25
SEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-M-P3' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#Las Palmas
EPOCHS=500
DATASET='LP'
DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
CHANN=128
SEED=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P3' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#########################################################àà
##PSIZE = 7
PATCHSIZE=7

#Madrid
EPOCHS=250
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25
SEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-M-P7' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#Las Palmas
EPOCHS=500
DATASET='LP'
DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
CHANN=128
SEED=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P7' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#########################################################àà
##PSIZE = 9
PATCHSIZE=9

#Madrid
EPOCHS=250
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25
SEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-M-P9' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#Las Palmas
EPOCHS=500
DATASET='LP'
DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
CHANN=128
SEED=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P9' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#########################################################àà
##PSIZE = 11
PATCHSIZE=11

#Madrid
EPOCHS=250
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25
SEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-M-P11' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#Las Palmas
EPOCHS=500
DATASET='LP'
DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
CHANN=128
SEED=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P11' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#########################################################àà
##PSIZE = 1
PATCHSIZE=1

#Madrid
EPOCHS=250
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25
SEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-M-P1' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

#Las Palmas
EPOCHS=500
DATASET='LP'
DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
CHANN=128
SEED=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model MamTrans --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'MamTrans-LP-P1' --comment 'MamTrans training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED

