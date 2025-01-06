import argparse
from datetime import datetime, timedelta
import time
import json
import os
import random
import warnings

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed
from torch.utils.data import TensorDataset
import torch.distributed as dist
import torch.nn as nn

from timm.scheduler import create_scheduler, CosineLRScheduler
from timm.optim import create_optimizer

from utils.LARC import LARC
from utils.focal import FocalLoss

import numpy as np

from models.ViT import ViT
from models.vim.models_mamba import VisionMamba
from models.HSIMamba import HSIClassificationMambaModel
from models.HiT import HiT, ConvPermuteMLP
from models.mamtrans.MamTrans import MamTrans
from models.ssmamba.ssmamba import mamba_SS_model
from models.CNN_2D import CNxtN_2D
from models.DBDA import DBDA
from models.SpectralFormer import SpectralFormer
from models.SSAN import SSAN
from models.GhostNet import GhostNet
from models.Hybrid3D_2D import Hyb3D_2D
from models.RSSAN import RSSAN
from models.LiteDepthwiseNet import LiteDwNet

import models.extraLayers

from engine import train_epoch, evaluate, test_evaluate
import utils.tools as tools

import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow

warnings.filterwarnings("ignore", category=UserWarning, module='torch.optim.lr_scheduler') #suppressing warning due to lr_scheduler accing current lr
warnings.filterwarnings("ignore", category=DeprecationWarning)

#mlflow.set_tracking_uri(Path(f"/home/{os.getenv("USER")}/mlruns").as_uri())

def get_args_parser():
    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)

    # Basic parameters
    parser.add_argument('--model-type', default='ViT', type=str, help='Model type (default: "ViT")')
    parser.add_argument('--batch-size', default=512, type=int, help='Batch size') #8192 to be used with LARS
    parser.add_argument('--epochs', default=1, type=int, help='Total epochs to run')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # ViT parameters
    parser.add_argument('--mlp-dim', default=4, type=int, help='Number of features in the mlp')
    parser.add_argument('--heads', default=16, type=int, help='Number of heads in the attention layers')
    parser.add_argument('--easyAtt', default=False, type=bool, help='Use EasyAttention instead of Attention')
    parser.add_argument('--caf', default=False, type=bool, help='Use CAF')
    parser.add_argument('--dropPath-rate', default=0.1, type=float, help='DropPath rate')

    # ViT/ViM parameters
    parser.add_argument('--blocks', default=4, type=int, help='Number of blocks in the transformer')
    parser.add_argument('--patch-size', default=9, type=int, help='Patch size')
    parser.add_argument('--embed-dim', default=64, type=int, help='Embeddings dimension')
    parser.add_argument('--classes', default=4, type=int, help='Number of classes to predict (default: 4)')
    parser.add_argument('--drop', type=float, default=0.2, metavar='PCT', help='Dropout rate (default: 0.1)')

    # HSIMamba parameters
    parser.add_argument('--deltat', default=0.01, type=float, help='Delta parameter for HSIMamba')
    parser.add_argument('--output-dim', default=128, type=int, help='Output dimension of the HSIMamba model')

    # HiTMamba parameters
    parser.add_argument('--large-features', action='store_true', default=False, help='Use a higher number of features')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--use-larc', default=True, type=bool, help='Use LARC')
    parser.add_argument('--criterion', default='focal', type=str, help='Criterion (default: "focal")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight-decay', type=float, default=5e-5, help='weight decay (default: 5e-5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

    # Focal loss parameters
    parser.add_argument('--gamma', type=float, default=2, help='Gamma parameter for focal loss (default: 2)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--t-initial', type=int, default=50, help='Initial T value for cosine scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--cycle-mul', type=float, default=1.3, metavar='N', help='cycle multiplier for cosine restarts')
    parser.add_argument('--cycle-limit', type=int, default=7, metavar='N', help='cycle limit for cosine restarts')
    parser.add_argument('--cycle-decay', type=int, default=0.9, metavar='N', help='cycle decay for cosine restarts')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    
    # Dataset parameters
    parser.add_argument('--db-name', default='madrid', type=str, help='dataset name')
    parser.add_argument('--data-path', default='/home/domenico/Desktop/test_modelli/datasets/Madrid/hsi/', type=str, help='dataset path') #/home/domenico/Desktop/dataset_experiments/CUBES_cal_alt/
    parser.add_argument('--gt-path', default='/home/domenico/Desktop/test_modelli/datasets/Madrid/gt/', type=str, help='dataset path') #/home/domenico/Desktop/dataset_experiments/HSI_GT/npyFiles/
    parser.add_argument('--channels', type=int, default=25, help='Number of channels in the dataset')
    parser.add_argument('--train-pcg', default='0.7', type=float, help='Train set split percentage')
    parser.add_argument('--val-pcg', default='0.1', type=float, help='Validation set split percentage')
    parser.add_argument('--densify-labels', default=[2,3], nargs='+', type=int, help="Labels to densify")
    parser.add_argument('--augment-labels', default=[2,3], nargs='+', type=int, help="Labels to augment")
    parser.add_argument('--weighted-sampler', action='store_true', default=False, help='Use a weighted sampler')

    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')

    # Mlflow parameters
    parser.add_argument('--sys-metrics', default=False, type=bool, help='Log system metrics')

    # Other parameters
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args):
    tools.init_distributed_mode(args)
    temp_dir = Path(f"./tmp")

    experiment_name = args.model_type
    experiment_description = f'{args.model_type} for brain tumor classification'
    run_name = f'{args.job_name}_{args.model_type}-{args.db_name}-{args.patch_size}-run-{datetime.now().strftime("%Y%m%d_%H%M%S")}' #args.job_name if run with submitit
    run_description = f'Analyze the behavior of the {args.model_type} using a recent version of the {args.db_name} HSI dataset.'

    #log
    if tools.is_main_process():
        print(args)
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.set_experiment_tag('mlflow.note.content', experiment_description)

        temp_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    if args.distributed:
        args.batch_size = int(args.batch_size / (args.ngpus*args.nodes))
        args.num_workers = int((args.num_workers + (args.ngpus*args.nodes) - 1) / (args.ngpus*args.nodes))

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    torch.cuda.empty_cache()


    with open(f'image_list_{args.db_name}.json', 'r') as f:
        image_list = json.load(f)

        
    """ RANDOM SPLITTING """
    train_val_ids = []
    tumor_IDs, nontumor_IDs = tools.get_tumor_IDs(image_list, args.gt_path)
    
    random.Random(seed).shuffle(tumor_IDs)
    random.Random(seed).shuffle(nontumor_IDs)

    T_train_ids, T_val_ids, T_test_ids = tools.random_split(tumor_IDs, args.train_pcg,
                                                            args.val_pcg, seed)
    
    train_ids, validation_ids, test_ids = tools.random_split(nontumor_IDs, args.train_pcg,
                                                             args.val_pcg, seed)

    train_ids.extend(T_train_ids)
    validation_ids.extend(T_val_ids)
    test_ids.extend(T_test_ids)

    train_val_ids.extend(train_ids)
    train_val_ids.extend(validation_ids)
    """ ********* """
      
    min_vect, max_vect = tools.min_max_norm_val(args.data_path, args.gt_path, train_val_ids, args.channels)

    train_data, train_labels, train_lab_count_noDens, _  = tools.loadImagesData(args.data_path, args.gt_path, train_ids, patch_size=args.patch_size, labelsToDensify=args.densify_labels, labelsToAugment=args.augment_labels, minMaxVects=[min_vect, max_vect])
    val_data, val_labels, val_lab_count_noDens, _ = tools.loadImagesData(args.data_path, args.gt_path, validation_ids, patch_size=args.patch_size, labelsToDensify=[], labelsToAugment=[], minMaxVects=[min_vect, max_vect])

    counts = train_lab_count_noDens + val_lab_count_noDens
    #unique, counts = np.unique(fnp.concatenate((train_lab_count_noDens, val_lab_count_noDens)), return_counts=True)

    raw_weights = {int(i): sum(counts) / count for i, count in enumerate(counts)}

    #weights normalization
    class_weights = {cls: weight / sum(raw_weights.values()) for cls, weight in raw_weights.items()}

    weights = [class_weights[i] for i in range(len(class_weights))]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    train_data=torch.from_numpy(train_data).type(torch.FloatTensor)
    train_labels=torch.from_numpy(train_labels).type(torch.LongTensor)
    val_data=torch.from_numpy(val_data).type(torch.FloatTensor)
    val_labels=torch.from_numpy(val_labels).type(torch.LongTensor)

    dataset_train = TensorDataset(train_data,train_labels)
    dataset_val = TensorDataset(val_data,val_labels)

    if args.distributed:
        num_tasks = tools.get_world_size()
        global_rank = tools.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.weighted_sampler:
            sampler_train = torch.utils.data.WeightedRandomSampler(class_weights_tensor, len(dataset_train), replacement=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if(args.model_type == 'ViT'):
        model = ViT(patchSize=args.patch_size, nBlocks=args.blocks, mlp_dim=args.mlp_dim, caf=args.caf, easyAtt=args.easyAtt, numHeads=args.heads, embedDim=args.embed_dim, numClasses=args.classes, dropout=args.drop, dropPath=args.dropPath_rate, channels=args.channels)
    elif args.model_type == 'DBDA':
      model = DBDA(band=args.channels, classes=args.classes)
    elif args.model_type == 'SpectralFormer':
      model = SpectralFormer(patch_size=args.patch_size, num_patches=args.channels, num_classes=args.classes)
    elif args.model_type == 'SSAN':
      model = SSAN(patch_size=args.patch_size, num_band=args.channels, n_classes=args.classes)
    elif args.model_type == 'GhostNet':
      model = GhostNet(input_channels=args.channels, n_classes=args.classes)
    elif args.model_type == 'Hyb3D_2D':
      model = Hyb3D_2D(in_chns=args.channels, patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'RSSAN':
      model = RSSAN(in_chns=args.channels, patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'LiteDwNet':
      model = LiteDwNet(in_chns=args.channels, patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'ViM':
        model = VisionMamba(patch_size=args.patch_size, num_classes=args.classes, embed_dim=args.embed_dim, depth=args.blocks, drop_rate=args.drop, channels=args.channels, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=False)
        model.patch_embed = models.extraLayers.PatchEmbedding(args.patch_size, args.embed_dim) #changes the patchembedding layer to adapt to the input format
    elif args.model_type == 'HSIMamba':
        model = nn.Sequential(
            models.extraLayers.PermuteLayer(0,2,3,1), #adapt the patch to the required input format (B, H, W, C)
            HSIClassificationMambaModel(spatial_dim=args.patch_size, num_bands=args.channels, hidden_dim=args.embed_dim, output_dim=args.output_dim, delta_param_init=args.deltat, num_classes=args.classes)
        )
    elif args.model_type == 'MamTrans':
        model = MamTrans(channels=args.channels, num_classes=args.classes, image_size=args.patch_size, emb_dim=args.embed_dim, num_heads=args.heads, num_layers=args.blocks, datasetname=args.db_name) #head_dim, hidden_dim
    elif args.model_type == 'SSMamba':
        model = mamba_SS_model(spa_img_size=(args.patch_size, args.patch_size), spe_img_size=(3,3), spa_patch_size=3, spe_patch_size=2, in_chans=args.channels, hid_chans = args.embed_dim, embed_dim=args.embed_dim, nclass=args.classes, drop_path=args.dropPath_rate, depth=args.blocks, bi=True, 
                                norm_layer=nn.LayerNorm, global_pool=False, cls = True, fu=True)
    elif args.model_type == 'HiT':
        if args.db_name == 'madrid':
            if args.large_features:
                embed_dims = [128, 128, 256, 256]
            else:
                embed_dims = [56, 56, 88, 88]
        elif args.db_name == 'LP':
            if args.large_features:
                embed_dims = [536, 536, 640, 640]
            else:
                embed_dims = [256, 256, 512, 512]
        model = nn.Sequential(
            models.extraLayers.AddDimensionLayer(1),
            HiT([4,3,14,3], img_size=args.patch_size, patch_size=3, in_chans=args.channels, num_classes=4,
                 embed_dims=embed_dims, transitions=[False, True, False, False], segment_dim=[8,8,4,4], mlp_ratios=[3,3,3,3], skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, mlp_fn=ConvPermuteMLP, large_features=args.large_features) #Doesn't work for single pixels, only for patches to capturre spatial information
        )
    else:
        print('Model not found')
        exit()

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    _optimizer = create_optimizer(args, model_without_ddp)
    if args.use_larc == True:
        optimizer = LARC(_optimizer)

    if args.sched == 'cosine_restart':
        lr_scheduler = CosineLRScheduler(_optimizer, t_initial=args.t_initial, cycle_limit=args.cycle_limit, cycle_mul=args.cycle_mul, k_decay=args.decay_rate, lr_min=args.min_lr, warmup_t=args.warmup_epochs, warmup_lr_init=args.warmup_lr)
    else:
        lr_scheduler, _ = create_scheduler(args, _optimizer)

    if args.criterion == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.criterion == 'weighted_cross_entropy':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)   
    elif args.criterion == 'focal':
        criterion = FocalLoss(alpha=None, reduction='mean', gamma=args.gamma, weight=class_weights_tensor)
    else:
        print('Criterion not found')
        exit()
        
    # log
    if tools.is_main_process():
        mlflow.start_run(log_system_metrics=args.sys_metrics, run_name=run_name, description=run_description)
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        mlflow.log_param("n_parameters", n_parameters)
        best_val_loss = float('inf')
        start_time = time.time()

    if args.distributed:
        dist.barrier()

    #TRAINING
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

            if args.dist_eval:
                data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_epoch(model, data_loader_train, optimizer, device, criterion, args)
        val_stats = evaluate(data_loader_val, model, device, criterion, args)
    
        if args.distributed:
            dist.barrier()

        # log
        training_log_stats = {**{f'training_{k}': v for k, v in train_stats.items()},
                    f'validation_avg_loss': val_stats["avg_loss"],
                    f'learningRate': _optimizer.param_groups[0]['lr']}
        validation_metrics_stats = {**{f'validation_{k}': v for k, v in val_stats.items() if k != 'cm' and k != 'avg_loss'}}
    
        if tools.is_main_process():
            plt.figure(figsize=(12, 12))
            sns.heatmap(val_stats["cm"], annot=True, fmt="d")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(temp_dir,f"validation_confusion_matrix.png"))
            plt.close()

            if(val_stats["avg_loss"] < best_val_loss):
                best_val_loss = val_stats["avg_loss"]
                if args.distributed:
                    model_to_save = model.module
                else:
                    model_to_save = model
                torch.save(model_to_save.state_dict(), os.path.join(temp_dir,f'{args.model_type}_best_model_{run_name}.pth'))

            mlflow.log_artifact(os.path.join(temp_dir,f"validation_confusion_matrix.png"), run_name)

            mlflow.log_metrics(training_log_stats, epoch)
            mlflow.log_metrics(validation_metrics_stats, epoch)

        #to next epoch
        if args.sched == 'plateau':
            lr_scheduler.step(val_stats["avg_loss"])
        elif args.sched == 'cosine' or args.sched == 'cosine_restart':
            lr_scheduler.step(epoch)
        else:
            print('Scheduler not found')
            exit()

    # log and register model
    if tools.is_main_process():
        if args.distributed:
            model_to_save = model.module
        else:
            model_to_save = model
        X_sample = torch.randn(1, args.channels, args.patch_size, args.patch_size).to(device)
        y_sample = model(X_sample)
        signature = mlflow.models.signature.infer_signature(X_sample.cpu().numpy(), y_sample.cpu().detach().numpy())

        mlflow.pytorch.log_model(model_to_save, f'{args.model_type}_best_model_{run_name}', signature=signature, registered_model_name=f'best_model_{run_name}')

        total_time = time.time() - start_time
        mlflow.log_param('total_training_time',  total_time)

    if args.distributed:
        dist.barrier()

    if(args.model_type == 'ViT'):
        model = ViT(patchSize=args.patch_size, nBlocks=args.blocks, mlp_dim=args.mlp_dim, caf=args.caf, easyAtt=args.easyAtt, numHeads=args.heads, embedDim=args.embed_dim, numClasses=args.classes, dropout=args.drop, dropPath=args.dropPath_rate, channels=args.channels)
    elif args.model_type == 'DBDA':
      model = DBDA(band=args.channels, classes=args.classes)
    elif args.model_type == 'SpectralFormer':
      model = SpectralFormer(patch_size=args.patch_size, num_patches=args.channels, num_classes=args.classes)
    elif args.model_type == 'SSAN':
      model = SSAN(patch_size=args.patch_size, num_band=args.channels, n_classes=args.classes)
    elif args.model_type == 'GhostNet':
      model = GhostNet(input_channels=args.channels, n_classes=args.classes)
    elif args.model_type == 'Hyb3D_2D':
      model = Hyb3D_2D(in_chns=args.channels, patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'RSSAN':
      model = RSSAN(in_chns=args.channels, patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'LiteDwNet':
      model = LiteDwNet(in_chns=args.channels, patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'ViM':
        model = VisionMamba(patch_size=args.patch_size, num_classes=args.classes, embed_dim=args.embed_dim, depth=args.blocks, drop_rate=args.drop, channels=args.channels, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True)
        model.patch_embed = models.extraLayers.PatchEmbedding(args.patch_size, args.embed_dim)
    elif args.model_type == 'HSIMamba':
        model = nn.Sequential(
            models.extraLayers.PermuteLayer(0,2,3,1),
            HSIClassificationMambaModel(spatial_dim=args.patch_size, num_bands=args.channels, hidden_dim=args.embed_dim, output_dim=args.output_dim, delta_param_init=args.deltat, num_classes=args.classes)
        )
    elif args.model_type == 'MamTrans':
        model = MamTrans(channels=args.channels, num_classes=args.classes, image_size=args.patch_size, emb_dim = args.embed_dim, num_heads=args.heads, num_layers=args.blocks, datasetname=args.db_name) #head_dim, hidden_dim
    elif args.model_type == 'SSMamba':
        model = mamba_SS_model(spa_img_size=(args.patch_size, args.patch_size), spe_img_size=(3,3), spa_patch_size=3, spe_patch_size=2, in_chans=args.channels, hid_chans = args.embed_dim, embed_dim=args.embed_dim, nclass=args.classes, drop_path=args.dropPath_rate, depth=args.blocks, bi=True, 
                                norm_layer=nn.LayerNorm, global_pool=False, cls = True, fu=True)
    elif args.model_type == 'HiT':
        if args.db_name == 'madrid':
            if args.large_features:
                embed_dims = [128, 128, 256, 256]
            else:
                embed_dims = [56, 56, 88, 88]
        elif args.db_name == 'LP':
            if args.large_features:
                embed_dims = [536, 536, 640, 640]
            else:
                embed_dims = [256, 256, 512, 512]
        model = nn.Sequential(
            models.extraLayers.AddDimensionLayer(1),
            HiT([4,3,14,3], img_size=args.patch_size, patch_size=3, in_chans=args.channels, num_classes=4,
                embed_dims=embed_dims, transitions=[False, True, False, False], segment_dim=[8,8,4,4], mlp_ratios=[3,3,3,3], skip_lam=1.0,
                qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, mlp_fn=ConvPermuteMLP, large_features=args.large_features)
        )
    else:
        print('Model not found')
        exit()

    model.load_state_dict(torch.load(os.path.join(temp_dir,f'{args.model_type}_best_model_{run_name}.pth'), weights_only=True))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    #TESTING
    for test_image in test_ids:
        hsi, gt, [height, width] = tools.get_cube_and_GT(test_image, args.data_path, args.gt_path, patch_size=args.patch_size, minMaxVects=[min_vect, max_vect])

        hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
        gt = torch.from_numpy(gt).type(torch.LongTensor)

        dataset_test = TensorDataset(hsi,gt)

        if args.dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=tools.get_world_size(), rank=tools.get_rank(), shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        
        test_preds, test_stats = test_evaluate(data_loader_test, model, device, args)
        test_preds_softmax = test_preds.numpy()
        test_preds_argmax = np.argmax(test_preds_softmax, axis=1)

        if args.distributed:
            dist.barrier()

        # log + final image generation
        if tools.is_main_process():
            for key, value in test_stats.items():
                if isinstance(value, np.ndarray):
                    test_stats[key] = value.tolist()  # convert NumPy array to list
            with open(os.path.join(temp_dir,f'{test_image}_test_metrics.json'), 'w') as json_file:
                json.dump(test_stats, json_file, indent=4)

            mlflow.log_artifact(os.path.join(temp_dir,f'{test_image}_test_metrics.json'), run_name)

            data_reshaped_argmax = np.reshape(test_preds_argmax, (height, width))
            data_reshaped_softmax = np.reshape(test_preds_softmax, (height, width, args.classes))

            npimg = tools.getImage(data_reshaped_argmax, height, width)
            npimg_prob = tools.getImageProb(data_reshaped_softmax, height, width)                                     

            plt.figure()
            plt.imshow(npimg)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(temp_dir,f'{run_name}_{test_image}.png'))
            plt.close()

            plt.figure()
            plt.imshow(npimg_prob)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(temp_dir,f'{run_name}_{test_image}_prob.png'))
            plt.close()

            mlflow.log_artifact(os.path.join(temp_dir,f'{run_name}_{test_image}.png'), run_name)
            mlflow.log_artifact(os.path.join(temp_dir,f'{run_name}_{test_image}_prob.png'), run_name)


    if tools.is_main_process():
        print(f"----------> Model {args.model_type} finished run")
        mlflow.end_run()

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()        

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()

	main(args)