"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os

import torch
import torch.distributed as dist

import cv2

import numpy as np

def augment_patches(samps, labs):
	if len(labs.shape)>2:
	
		lr_transf = samps[:,::-1,...]
		ud_transf = samps[:,:,::-1,:]
		samps_r90 = np.rot90(samps, 1, (1,2))
		samps_r180 = np.rot90(samps_r90, 1, (1,2))
		samps_r270 = np.rot90(samps_r180, 2, (1,2))

		aug_samps = np.vstack((lr_transf, ud_transf, samps_r90,
							samps_r180, samps_r270, samps))


		lab_lr_transf = labs[:,::-1,...]
		lab_ud_transf = labs[:,:,::-1]
		labsr90 = np.rot90(labs, 1, (1,2))
		labsr180 = np.rot90(labsr90, 1, (1,2))
		labsr270 = np.rot90(labsr180, 2, (1,2))

		aug_labs = np.vstack((lab_lr_transf, lab_ud_transf, labsr90, labsr180, labsr270, labs))
	else:
		if len(samps.shape)>2:
			
			lr_transf = samps[:,::-1,...]
			ud_transf = samps[:,:,::-1,:]
			samps_r90 = np.rot90(samps, 1, (1,2))
			samps_r180 = np.rot90(samps_r90, 1, (1,2))
			samps_r270 = np.rot90(samps_r180, 2, (1,2))

			aug_samps = np.vstack((lr_transf, ud_transf, samps_r90, samps_r180, samps_r270, samps))
		else:
			aug_samps = np.tile(samps, (6,1))
		
		aug_labs = np.tile(labs, 6)
		
	return aug_samps, aug_labs

def getImageProb(data, height, width):
    colormap = np.array([
        [0,255,0],
        [255,0,0],
        [0,0,255],
        [255,182,193]
    ])

    npimg = np.zeros((height, width, 3))

    for class_index in range(4):
        npimg += data[:, :, class_index][:, :, np.newaxis] * colormap[class_index]

    return npimg.astype('uint8')

def getImage(data, height, width):
    colorIndex = {0:[0,255,0],1:[255,0,0],2:[0,0,255],3:[255,182,193]}

    npimg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            npimg[y, x, :] = colorIndex.get(data[y][x], [0, 0, 0])
    return npimg

def min_max_norm_val(hsi_path, gt_path, imageList, channels):
    samples = []
    
    for imgID in imageList:
        hsi_dataset= np.load(hsi_path+imgID+'.npy')
        labels = np.load(gt_path+imgID+'.npy')
		# gt = np.array(Image.open(f'{gt_path}{idp}.png'), dtype=np.uint8)

        #hsi_dataset = smooth_cube(hsi_dataset)
        samples.append(hsi_dataset[labels>0,:])

    train_set = np.vstack(samples)

	# train_set = SG_filter(train_set)

    min_vect = np.amin(train_set, axis=0).reshape(1, 1, channels)
    max_vect = np.amax(train_set, axis=0).reshape(1, 1, channels)

    return min_vect, max_vect

def get_cube_and_GT(idp, data_path, gt_path, patch_size, minMaxVects):
    data = np.load(f"{data_path}{idp}.npy")
    gt = np.load(f"{gt_path}{idp}.npy")
	# gt = np.array(Image.open(f'{gt_path}{idp}.png'), dtype=np.uint8)

	# data = smooth_cube(data)
	# data = SG_filter(data)
    data = data-minMaxVects[0]/(minMaxVects[1]-minMaxVects[0])

    H,W,C = data.shape
    if patch_size > 1:
        hsi_data = np.pad(data, ((patch_size//2,patch_size//2),
            (patch_size//2,patch_size//2),(0,0)), "constant")

        Y, X = np.mgrid[0:H, 0:W]
        c_yx = np.stack((Y.ravel(), X.ravel()),axis=1)+patch_size//2

        y_min = c_yx[:,0]-patch_size//2
        y_max = c_yx[:,0]+1+patch_size//2
        x_min = c_yx[:,1]-patch_size//2
        x_max = c_yx[:,1]+1+patch_size//2

        y_grid, x_grid = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')

        data = hsi_data[y_min[:, np.newaxis, np.newaxis] + y_grid, 
                        x_min[:, np.newaxis, np.newaxis] + x_grid, :]
    else:
        data = data.reshape(H*W, C)
        data = data[:, np.newaxis, np.newaxis, :]
			
    samps = np.moveaxis(data, -1, 1)
    gt = gt.reshape(H*W)
        
    return samps, gt, [H,W]

def loadImagesData(hsi_path, gt_path, imglist, patch_size, labelsToDensify, labelsToAugment, minMaxVects):
    data_samps = []
    gt_labs = []
         
    for imgID in imglist:
        hsi_dataset= np.load(hsi_path+imgID+'.npy')
        labels = np.load(gt_path+imgID+'.npy')

        if len(labelsToDensify) != 0:
            labels = densify_gt(labels, labelsToDensify)

        hsi_dataset = hsi_dataset-minMaxVects[0]/(minMaxVects[1]-minMaxVects[0]) #min-max normalization

        for l in np.unique(labels)[1:]: #labels 0 -> background
            if patch_size>1:
                label_mask = (labels==l).astype('uint8')
                c_yx = np.stack(np.where(label_mask>0), axis=1)

                hsi_dataset = np.pad(hsi_dataset, ((patch_size//2,patch_size//2),
                            (patch_size//2,patch_size//2),(0,0)), "constant")
                labels = np.pad(labels, ((patch_size//2,patch_size//2),
                            (patch_size//2,patch_size//2)), "constant")
						
                y_min = c_yx[:,0]-patch_size//2
                y_max = c_yx[:,0]+1+patch_size//2
                x_min = c_yx[:,1]-patch_size//2
                x_max = c_yx[:,1]+1+patch_size//2
					
                y_grid, x_grid = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')

                samps = hsi_dataset[y_min[:, np.newaxis, np.newaxis] + y_grid, 
                                    x_min[:, np.newaxis, np.newaxis] + x_grid, :]

            else:       
                samps = hsi_dataset[labels==l, np.newaxis, np.newaxis, :]

            lab = np.ones(samps.shape[0])*l

            if l in labelsToAugment:
                samps, lab = augment_patches(samps, lab)

            data_samps.append(samps)
            gt_labs.append(lab[:,None])

    data_samps = np.vstack(data_samps)
    data_samps = np.moveaxis(data_samps, -1, 1)
    gt_labs = np.vstack(gt_labs).astype(int)
    gt_labs = np.squeeze(gt_labs)

    return data_samps, gt_labs-1 #0: healthy, 1: tumor, 2: blood, 3: duraMater

def dilate(img, k=3):
	st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
	return cv2.dilate(img, st_elem)

def erode(img, k=3, ellipse=True):
	if ellipse:
		st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
	else:
		st_elem = np.ones((k,k), np.uint8)
	return cv2.erode(img, st_elem)

def densify_gt(gt, dens_labels):
	out_gt = np.zeros_like(gt)

	for l in np.unique(gt)[1:]:
		label_mask = (gt==l).astype('uint8')

		if l in dens_labels:
			label_mask = erode(dilate(label_mask, k=3), k=3)
		
		out_gt[label_mask>0] = label_mask[label_mask>0]*l

	return out_gt

def gather_tensor(tensor):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    return dist.get_world_size()


def get_rank():
    if torch.distributed.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)