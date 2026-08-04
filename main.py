import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from mlflow.tracking import MlflowClient
from timm.optim import create_optimizer
from timm.scheduler import CosineLRScheduler, create_scheduler
from torch.utils.data import TensorDataset

import models.extraLayers
import utils.tools as tools
from engine import evaluate, test_evaluate, train_epoch
from models.Conv.GhostNet import GhostNet
# from models.Conv.CNN_2D import CNN_2D
from models.DBDA import DBDA
from models.EfficientNet import EfficientNet
from models.HiT import ConvPermuteMLP, HiT
from models.HSIMamba import HSIClassificationMambaModel
from models.Hybrid3D_2D import Hyb3D_2D
from models.LiteDepthwiseNet import LiteDwNet
from models.mamtrans.MamTrans import MamTrans
from models.RSSAN import RSSAN
from models.SpectralFormer import SpectralFormer
from models.SSAN import SSAN
from models.ssmamba.ssmamba import mamba_SS_model
from models.vim.models_mamba import VisionMamba
from models.ViT import ViT
from utils.focal import FocalLoss
from utils.LARC import LARC

matplotlib.use('Agg')


# suppressing warning due to lr_scheduler accing current lr
warnings.filterwarnings("ignore", category=UserWarning,
                        module='torch.optim.lr_scheduler')
warnings.filterwarnings("ignore", category=DeprecationWarning)
exclude_params = ['inference', 'run_id', 'tracking_uri', 'device', 'data_path', 'gt_path', 'weighted_sampler',
                  'sys_metrics', 'num_workers', 'pin_mem', 'distributed', 'world_size', 'dist_eval', 'ngpus', 'nodes']


def get_args_parser():

    parser = argparse.ArgumentParser(
        'Training and evaluation script', add_help=False)

    parser.add_argument('--inference', action='store_true',
                        default=False, help='Perform inference only - no training')
    parser.add_argument('--run-id', default='4b5f59d9505a466bb6fd381b2fe993e1',
                        type=str, help='MLFlow Run ID to be used for inference')
    parser.add_argument('--tracking-uri',
                        default="file://" + os.getcwd() + "/mlruns_final_complete",
                        type=str, help='MLFlow tracking URI'
                        )

    # Basic parameters
    parser.add_argument('--model-type', default='MamTrans',
                        type=str, help='Model type (default: "ViT")')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size')  # 8192 to be used with LARS
    parser.add_argument('--epochs', default=1, type=int,
                        help='Total epochs to run')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # ViT parameters
    parser.add_argument('--mlp-dim', default=4, type=int,
                        help='Number of features in the mlp')
    parser.add_argument('--heads', default=16, type=int,
                        help='Number of heads in the attention layers')
    parser.add_argument('--easyAtt', default=False, type=bool,
                        help='Use EasyAttention instead of Attention')
    parser.add_argument('--caf', default=False, type=bool, help='Use CAF')
    parser.add_argument('--dropPath-rate', default=0.1,
                        type=float, help='DropPath rate')

    # ViT/ViM parameters
    parser.add_argument('--blocks', default=4, type=int,
                        help='Number of blocks in the transformer')
    parser.add_argument('--patch-size', default=9, type=int, help='Patch size')
    parser.add_argument('--embed-dim', default=64,
                        type=int, help='Embeddings dimension')
    parser.add_argument('--classes', default=4, type=int,
                        help='Number of classes to predict (default: 4)')
    parser.add_argument('--drop', type=float, default=0.2,
                        metavar='PCT', help='Dropout rate (default: 0.1)')

    # HSIMamba parameters
    parser.add_argument('--deltat', default=0.01, type=float,
                        help='Delta parameter for HSIMamba')
    parser.add_argument('--output-dim', default=128, type=int,
                        help='Output dimension of the HSIMamba model')

    # HiTMamba parameters
    parser.add_argument('--large-features', action='store_true',
                        default=False, help='Use a higher number of features')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str,
                        metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--use-larc', default=True, type=bool, help='Use LARC')
    parser.add_argument('--criterion', default='focal',
                        type=str, help='Criterion (default: "focal")')
    parser.add_argument('--opt-eps', default=1e-8, type=float,
                        metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+',
                        metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight-decay', type=float,
                        default=5e-5, help='weight decay (default: 5e-5)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='SGD momentum (default: 0.9)')

    # Focal loss parameters
    parser.add_argument('--gamma', type=float, default=2,
                        help='Gamma parameter for focal loss (default: 2)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str,
                        metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3,
                        metavar='LR', help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None,
                        metavar='pct, pct', help='Learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='Learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0,
                        metavar='STDDEV', help='Learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6,
                        metavar='LR', help='Warmup learning rate (default: 1e-6)')
    parser.add_argument('--t-initial', type=int, default=50,
                        help='Initial T value for cosine scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=10,
                        metavar='N', help='Epoch interval to decay LR')
    parser.add_argument('--cycle-mul', type=float, default=1.3,
                        metavar='N', help='Cycle multiplier for cosine restarts')
    parser.add_argument('--cycle-limit', type=int, default=7,
                        metavar='N', help='Cycle limit for cosine restarts')
    parser.add_argument('--cycle-decay', type=int, default=0.9,
                        metavar='N', help='Cycle decay for cosine restarts')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        metavar='N', help='Epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='Epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='Patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1,
                        metavar='RATE', help='LR decay rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--db-name', default='LP',
                        type=str, help='Dataset name')
    parser.add_argument('--data-path', default='/path/to/hsi_npy/',
                        type=str, help='Dataset path')
    parser.add_argument('--gt-path', default='/path/to/gt_map/',
                        type=str, help='Dataset path')
    parser.add_argument('--channels', type=int, default=128,
                        help='Number of channels in the dataset')
    parser.add_argument('--train-pcg', default='0.7',
                        type=float, help='Train set split percentage')
    parser.add_argument('--val-pcg', default='0.1', type=float,
                        help='Validation set split percentage')
    parser.add_argument(
        '--densify-labels', default=[2, 3], nargs='+', type=int, help="Labels to densify")
    parser.add_argument(
        '--augment-labels', default=[2, 3], nargs='+', type=int, help="Labels to augment")
    parser.add_argument('--weighted-sampler', action='store_true',
                        default=False, help='Use a weighted sampler')

    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true',
                        default=False, help='Enabling distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')

    # Mlflow parameters
    parser.add_argument('--sys-metrics', default=False,
                        type=bool, help='Log system metrics')

    # Other parameters
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false',
                        dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    return parser


def select_model(args):
    # By default the hyperparameters are the ones used for the original papers
    if (args.model_type == 'ViT'):
        model = ViT(patchSize=args.patch_size, nBlocks=args.blocks, mlp_dim=args.mlp_dim, caf=args.caf, easyAtt=args.easyAtt, numHeads=args.heads,
                    embedDim=args.embed_dim, numClasses=args.classes, dropout=args.drop, dropPath=args.dropPath_rate, channels=args.channels)
    elif args.model_type == 'EfficientNet':
        model = EfficientNet(in_dims=args.channels, out_classes=args.classes)
    elif args.model_type == 'DBDA':
        model = DBDA(band=args.channels, classes=args.classes)
    elif args.model_type == 'SpectralFormer':
        model = SpectralFormer(patch_size=args.patch_size,
                               num_patches=args.channels, num_classes=args.classes)
    elif args.model_type == 'SSAN':
        model = SSAN(patch_size=args.patch_size,
                     num_band=args.channels, n_classes=args.classes)
    elif args.model_type == 'GhostNet':
        model = GhostNet(input_channels=args.channels, n_classes=args.classes)
    elif args.model_type == 'Hyb3D_2D':
        model = Hyb3D_2D(in_chns=args.channels,
                         patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'RSSAN':
        model = RSSAN(in_chns=args.channels,
                      patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'LiteDwNet':
        model = LiteDwNet(in_chns=args.channels,
                          patch_size=args.patch_size, out_classes=args.classes)
    elif args.model_type == 'ViM':
        # embed_dim=args.embed_dim, depth=args.blocks, drop_rate=args.drop, ,
        # rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        # final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
        # if_rope_residual=False, bimamba_type="v2", if_cls_token=True,
        # if_divide_out=True, use_middle_cls_token=False
        model = VisionMamba(img_size=args.patch_size, patch_size=args.patch_size,
                            num_classes=args.classes, channels=args.channels, embed_dim=192, depth=5)
        # changes the patchembedding layer to adapt to the input format,
        # embed_dim = args.embed_dim
        model.patch_embed = models.extraLayers.PatchEmbedding(
            args.patch_size, embedDim=192)
    elif args.model_type == 'HSIMamba':
        model = nn.Sequential(
            # adapt the patch to the required input format (B, H, W, C)
            models.extraLayers.PermuteLayer(0, 2, 3, 1),
            HSIClassificationMambaModel(spatial_dim=args.patch_size, num_bands=args.channels,
                                        num_classes=args.classes, hidden_dim=256, output_dim=128, delta_param_init=0.01)
        )
    elif args.model_type == 'MamTrans':
        # head_dim, hidden_dim , emb_dim=args.embed_dim, num_heads=args.heads, num_layers=args.blocks,
        model = MamTrans(channels=args.channels, num_classes=args.classes,
                         image_size=args.patch_size, datasetname=None)
    elif args.model_type == 'SSMamba':
        model = mamba_SS_model(spa_img_size=(args.patch_size, args.patch_size), spe_img_size=(3, 3), spa_patch_size=3, spe_patch_size=2, in_chans=args.channels, nclass=args.classes,
                               hid_chans=64, embed_dim=64, global_pool=True)
    elif args.model_type == 'HiT':
        if args.db_name == 'Madrid':
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
            HiT(layers=[4, 3, 14, 3], img_size=args.patch_size, in_chans=args.channels, num_classes=args.classes,
                embed_dims=embed_dims, transitions=[False, True, False, False], segment_dim=[8, 8, 4, 4], mlp_ratios=[3, 3, 3, 3], skip_lam=1.0,
                qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                # Doesn't work for single pixels, only for patches to capturre
                # spatial information
                norm_layer=nn.LayerNorm, mlp_fn=ConvPermuteMLP, large_features=args.large_features)
        )
    else:
        print('Model not found')
        exit()

    return model


def main(args):
    os.environ['MLFLOW_TRACKING_URI'] = args.tracking_uri
    # #TEST

    tools.init_distributed_mode(args)
    temp_dir = Path("./tmp")

    experiment_name = args.model_type
    experiment_description = f'{args.model_type} for brain tumor classification'
    # args.job_name if run with submitit
    run_name = f'{args.job_name}_{args.model_type}-{args.db_name}-'\
               f'{args.patch_size}-run-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_description = f'Analyze the behavior of the {args.model_type}'\
                      f'using a recent version of the {args.db_name} HSI dataset.'

    if args.inference:
        run_data = None
        if not args.run_id:
            print('Run ID not provided')
            exit()
        else:
            client = MlflowClient()
            experiments = client.search_experiments()
            run_found = False
            for experiment in experiments:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id])

                for run in runs:
                    if run.info.run_id == args.run_id:
                        run_found = True
                        break

                if run_found is True:
                    break

            if run_found is False:
                print(f'Run {args.run_id} not found')
                exit()
            else:
                run_data = client.get_run(args.run_id)
                mlflow_params = run_data.data.params

                for param, value in mlflow_params.items(
                ):  # update args with the ones saved in the run
                    if hasattr(args, param) and param not in exclude_params:
                        if value != 'None':
                            param_type = type(getattr(args, param))
                            setattr(args, param, param_type(value))

        run_name = run_data.info.run_name

    print("Run name:", run_name)

    # log
    if tools.is_main_process():
        print(args)
        if not args.inference:
            mlflow.set_experiment(experiment_name=experiment_name)
            mlflow.set_experiment_tag(
                'mlflow.note.content', experiment_description)

        temp_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    if args.distributed:
        args.batch_size = int(args.batch_size / (args.ngpus * args.nodes))
        args.num_workers = int(
            (args.num_workers + (args.ngpus * args.nodes) - 1) / (args.ngpus * args.nodes))

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

    print("Train set:", train_ids)
    print("Validation set:", validation_ids)
    print("Test set:", test_ids)

    train_val_ids.extend(train_ids)
    train_val_ids.extend(validation_ids)
    """ ********* """

    min_vect, max_vect = tools.min_max_norm_val(
        args.data_path, args.gt_path, train_val_ids, args.channels)

    if not args.inference:
        train_data, train_labels, train_lab_count_noDens, _ = tools.loadImagesData(
            args.data_path, args.gt_path, train_ids, patch_size=args.patch_size, labelsToDensify=args.densify_labels, labelsToAugment=args.augment_labels, minMaxVects=[min_vect, max_vect])
        val_data, val_labels, val_lab_count_noDens, _ = tools.loadImagesData(
            args.data_path, args.gt_path, validation_ids, patch_size=args.patch_size, labelsToDensify=[], labelsToAugment=[], minMaxVects=[min_vect, max_vect])

        counts = train_lab_count_noDens + val_lab_count_noDens
        # unique, counts = np.unique(fnp.concatenate((train_lab_count_noDens, val_lab_count_noDens)), return_counts=True)

        raw_weights = {int(i): sum(counts) / count for i,
                       count in enumerate(counts)}

        # weights normalization
        class_weights = {
            cls: weight / sum(raw_weights.values()) for cls, weight in raw_weights.items()}

        weights = [class_weights[i] for i in range(len(class_weights))]
        class_weights_tensor = torch.tensor(
            weights, dtype=torch.float32, device=device)

        train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
        val_data = torch.from_numpy(val_data).type(torch.FloatTensor)
        val_labels = torch.from_numpy(val_labels).type(torch.LongTensor)

        dataset_train = TensorDataset(train_data, train_labels)
        dataset_val = TensorDataset(val_data, val_labels)

        if args.distributed:
            num_tasks = tools.get_world_size()
            global_rank = tools.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                          'This will slightly alter validation results as extra duplicate entries are added to achieve '
                          'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            if args.weighted_sampler:
                sampler_train = torch.utils.data.WeightedRandomSampler(
                    class_weights_tensor, len(dataset_train), replacement=True)
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

        if args.distributed:
            dist.barrier()

        model = select_model(args)
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            model_without_ddp = model.module
        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

        _optimizer = create_optimizer(args, model_without_ddp)
        if args.use_larc is True:
            optimizer = LARC(_optimizer)

        if args.sched == 'cosine_restart':
            lr_scheduler = CosineLRScheduler(_optimizer, t_initial=args.t_initial, cycle_limit=args.cycle_limit, cycle_mul=args.cycle_mul,
                                             k_decay=args.decay_rate, lr_min=args.min_lr, warmup_t=args.warmup_epochs, warmup_lr_init=args.warmup_lr)
        else:
            lr_scheduler, _ = create_scheduler(args, _optimizer)

        if args.criterion == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif args.criterion == 'weighted_cross_entropy':
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        elif args.criterion == 'focal':
            criterion = FocalLoss(
                alpha=None, reduction='mean', gamma=args.gamma, weight=class_weights_tensor)
        else:
            print('Criterion not found')
            exit()

        # log
        if tools.is_main_process():
            mlflow.start_run(log_system_metrics=args.sys_metrics,
                             run_name=run_name, description=run_description)
            for key, value in vars(args).items():
                mlflow.log_param(key, value)
            mlflow.log_param("n_parameters", n_parameters)
            best_val_loss = float('inf')
            start_time = time.time()

        if args.distributed:
            dist.barrier()

        # TRAINING
        for epoch in range(args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

                if args.dist_eval:
                    data_loader_val.sampler.set_epoch(epoch)

            train_stats = train_epoch(
                model, data_loader_train, optimizer, device, criterion, args)
            val_stats = evaluate(data_loader_val, model,
                                 device, criterion, args)

            if args.distributed:
                dist.barrier()

            # log
            training_log_stats = {**{f'training_{k}': v for k, v in train_stats.items()},
                                  'validation_avg_loss': val_stats["avg_loss"],
                                  'learningRate': _optimizer.param_groups[0]['lr']}
            validation_metrics_stats = {
                **{f'validation_{k}': v for k, v in val_stats.items() if k != 'cm' and k != 'avg_loss'}}

            if tools.is_main_process():
                plt.figure(figsize=(12, 12))
                sns.heatmap(val_stats["cm"], annot=True, fmt="d")
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.title('Confusion Matrix')
                plt.savefig(os.path.join(
                    temp_dir, "validation_confusion_matrix.png"), bbox_inches="tight", pad_inches=0, dpi=1200)
                plt.close()

                if (val_stats["avg_loss"] < best_val_loss):
                    best_val_loss = val_stats["avg_loss"]
                    if args.distributed:
                        model_to_save = model.module
                    else:
                        model_to_save = model
                    torch.save(model_to_save.state_dict(), os.path.join(
                        temp_dir, f'{args.model_type}_best_model_{run_name}.pth'))

                mlflow.log_artifact(os.path.join(
                    temp_dir, "validation_confusion_matrix.png"), run_name)

                mlflow.log_metrics(training_log_stats, epoch)
                mlflow.log_metrics(validation_metrics_stats, epoch)

            # to next epoch
            if args.sched == 'plateau':
                lr_scheduler.step(val_stats["avg_loss"])
            elif args.sched == 'cosine' or args.sched == 'cosine_restart':
                lr_scheduler.step(epoch)
            else:
                print('Scheduler not found')
                exit()

        # log and register model
        if tools.is_main_process():
            model = select_model(args)     
            model.load_state_dict(torch.load(os.path.join(temp_dir, f'{args.model_type}_best_model_{run_name}.pth')))
            model.to(device)    

            X_sample = torch.randn(2, args.channels, args.patch_size, args.patch_size).to(device)
            y_sample = model(X_sample)
            signature = mlflow.models.signature.infer_signature(
                X_sample.cpu().numpy(), y_sample.cpu().detach().numpy())

            mlflow.pytorch.log_model(model_to_save, f'{args.model_type}_best_model_{run_name}',
                                     signature=signature, registered_model_name=f'best_model_{run_name}')

            total_time = time.time() - start_time
            mlflow.log_param('total_training_time', total_time)

        if args.distributed:
            dist.barrier()

    # model = select_model(args)
    # model.load_state_dict(torch.load(client.download_artifacts(args.run_id, f'{args.model_type}_best_model_{run_name}.pth'), weights_only=True))
        del model
    model = mlflow.pytorch.load_model(f"models:/best_model_{run_name}/latest", map_location=torch.device("cuda:0"))  # load last registered model
    model.to(torch.device("cuda:0"))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # TESTING
    for test_image in test_ids:
        hsi, gt, [height, width] = tools.get_cube_and_GT(
            test_image, args.data_path, args.gt_path, patch_size=args.patch_size, minMaxVects=[min_vect, max_vect])

        hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
        gt = torch.from_numpy(gt.astype(np.int64)).type(torch.LongTensor)

        dataset_test = TensorDataset(hsi, gt)

        if args.dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=tools.get_world_size(), rank=tools.get_rank(), shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        test_preds, test_stats = test_evaluate(
            data_loader_test, model, device, args)
        test_preds_softmax = test_preds.numpy()
        test_preds_argmax = np.argmax(test_preds_softmax, axis=1)

        if args.distributed:
            dist.barrier()

        # log + final image generation
        if tools.is_main_process():
            for key, value in test_stats.items():
                if isinstance(value, np.ndarray):
                    test_stats[key] = value.tolist()
            with open(os.path.join(temp_dir, f'{test_image}_test_metrics.json'), 'w') as json_file:
                json.dump(test_stats, json_file, indent=4)

            data_reshaped_argmax = np.reshape(
                test_preds_argmax, (height, width))
            data_reshaped_softmax = np.reshape(
                test_preds_softmax, (height, width, args.classes))

            npimg = tools.getImage(data_reshaped_argmax, height, width)
            npimg_prob = tools.getImageProb(
                data_reshaped_softmax, height, width)

            plt.figure()
            plt.imshow(npimg)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
            plt.savefig(os.path.join(temp_dir, f'{run_name}_{test_image}.png'), bbox_inches="tight", pad_inches=0, dpi=1200)
            plt.close()

            plt.figure()
            plt.imshow(npimg_prob)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
            plt.savefig(os.path.join(
                temp_dir, f'{run_name}_{test_image}_prob.png'), bbox_inches="tight", pad_inches=0, dpi=1200)
            plt.close()

            if args.inference:
                with mlflow.start_run(run_id=args.run_id) as run:
                    mlflow.log_artifact(os.path.join(
                        temp_dir, f'{test_image}_test_metrics.json'), run_name)
                    mlflow.log_artifact(os.path.join(
                        temp_dir, f'{run_name}_{test_image}.png'), run_name)
                    mlflow.log_artifact(os.path.join(
                        temp_dir, f'{run_name}_{test_image}_prob.png'), run_name)
            else:
                mlflow.log_artifact(os.path.join(
                    temp_dir, f'{test_image}_test_metrics.json'))
                mlflow.log_artifact(os.path.join(
                    temp_dir, f'{run_name}_{test_image}.png'))
                mlflow.log_artifact(os.path.join(
                    temp_dir, f'{run_name}_{test_image}_prob.png'))

    if tools.is_main_process():
        print(f"----------> Model {args.model_type} finished run")
        mlflow.end_run()

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Training and Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
