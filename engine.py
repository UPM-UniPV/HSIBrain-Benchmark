"""
Train and eval functions
"""
from typing import Iterable

import torch
import torch.distributed as dist

import utils.tools as tools

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, cohen_kappa_score
import numpy as np

def calculate_metrics(preds, labels):
    val_preds_softmax = torch.softmax(preds, dim=1).numpy()
    val_preds_argmax = np.argmax(val_preds_softmax, axis=1)
    #al_preds_argmax = torch.argmax(preds, dim=1).cpu().numpy()
    val_labels = labels.numpy()

    correct = (val_preds_argmax == val_labels).sum().item()

    precision = precision_score(val_labels, val_preds_argmax, average='weighted', zero_division=0, labels=[0,1,2,3])
    recall = recall_score(val_labels, val_preds_argmax, average='weighted', zero_division=0, labels=[0,1,2,3])
    f1 = f1_score(val_labels, val_preds_argmax, average='weighted', zero_division=0, labels=[0,1,2,3])

    cm = confusion_matrix(val_labels, val_preds_argmax, labels=[0,1,2,3])

    accuracy = correct / len(val_labels)

    """ AUC """
    auc_class = np.array([-1.,-1.,-1.,-1.])
    roc_auc_list, weights = [], []
    n_samples = len(val_labels)

    healthy_gt = np.sum(val_labels==0)>0
    tumor_gt = np.sum(val_labels==1)>0
    vessel_gt = np.sum(val_labels==2)>0
    outer_gt = np.sum(val_labels==3)>0

    active_vect = np.where(np.array([healthy_gt, tumor_gt, vessel_gt, outer_gt]))[0]

    for l in active_vect:
        y_binary = (val_labels == l).astype(int)
        y_prob_l = val_preds_softmax[:, l]

		# Calculate ROC AUC for the current class
        roc_auc = roc_auc_score(y_binary, y_prob_l)
        roc_auc_list.append(roc_auc)

        # Calculate the weight for the current class
        weight = np.sum(y_binary) / n_samples
        weights.append(weight)

    auc_c = np.array(roc_auc_list)
    auc_wavg = np.sum(auc_c * np.array(weights))
    auc_class[active_vect] = auc_c

    kappa_score = cohen_kappa_score(val_labels, val_preds_argmax, labels=[0,1,2,3])

    return kappa_score, precision, recall, f1, accuracy, auc_wavg, auc_class, cm

def calculate_test_metrics(preds, labels):
    test_preds_softmax = torch.softmax(preds,dim=1).numpy()
    test_preds_argmax = np.argmax(test_preds_softmax, axis=1)
    #test_preds_argmax = torch.argmax(preds, dim=1).cpu().numpy()
    test_labels = labels.numpy()

    correct = (test_preds_argmax == test_labels).sum().item()
    cm = confusion_matrix(test_labels, test_preds_argmax, labels=[0,1,2,3])

    #overall
    accuracy = correct / len(test_labels)
    precision = precision_score(test_labels, test_preds_argmax, average='weighted', zero_division=0, labels=[0,1,2,3])
    recall = recall_score(test_labels, test_preds_argmax, average='weighted', zero_division=0, labels=[0,1,2,3])
    f1 = f1_score(test_labels, test_preds_argmax, average='weighted', zero_division=0, labels=[0,1,2,3])

    #roc per class + overall
    """ AUC """
    auc_class = np.array([-1.,-1.,-1.,-1.])
    roc_auc_list, weights = [], []
    n_samples = len(test_labels)

    healthy_gt = np.sum(test_labels==0)>0
    tumor_gt = np.sum(test_labels==1)>0
    vessel_gt = np.sum(test_labels==2)>0
    outer_gt = np.sum(test_labels==3)>0

    active_vect = np.where(np.array([healthy_gt, tumor_gt, vessel_gt, outer_gt]))[0]

    for l in active_vect:
        y_binary = (test_labels == l).astype(int)
        y_prob_l = test_preds_softmax[:, l]

		# Calculate ROC AUC for the current class
        roc_auc = roc_auc_score(y_binary, y_prob_l)
        roc_auc_list.append(roc_auc)

        # Calculate the weight for the current class
        weight = np.sum(y_binary) / n_samples
        weights.append(weight)

    auc_c = np.array(roc_auc_list)
    auc_wavg = np.sum(auc_c * np.array(weights))
    auc_class[active_vect] = auc_c

    #per class
    precision_class, recall_class, fscore_class, support = precision_recall_fscore_support(test_labels, test_preds_argmax, beta=1.0, average=None, zero_division=0, labels=[0,1,2,3])
    per_class_accuracy = np.where(cm.sum(axis=1) == 0, 0, cm.diagonal() / cm.sum(axis=1))

    kappa_score = cohen_kappa_score(test_labels, test_preds_argmax, labels=[0,1,2,3])

    return kappa_score, precision, recall, f1, accuracy, auc_wavg, cm, per_class_accuracy, precision_class, recall_class, fscore_class, auc_class, support

def train_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, criterion, args):
    
    model.train(True)
    running_loss = 0.0
    for _, (samples, targets) in enumerate(data_loader, 0):        
        samples = samples.to(dtype=torch.float32, device=device, non_blocking=True)
        targets = targets.to(dtype=torch.long, device=device, non_blocking=True)
        
        optimizer.zero_grad()

        torch.cuda.synchronize()

        outputs = model(samples)

        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()

        torch.cuda.synchronize()
        
        if args.distributed:
            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        
        running_loss += loss.item()
    
    avg_loss = running_loss /(len(data_loader)*args.world_size) #world_size = 1 if not distributed
    return {"avg_loss":avg_loss}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args):

    model.eval()
    val_preds = []
    val_labels = []
    running_loss = 0.0
    for _, (samples, targets) in enumerate(data_loader, 0):
        samples = samples.to(dtype=torch.float32, device=device, non_blocking=True)
        targets = targets.to(dtype=torch.long, device=device, non_blocking=True)

        torch.cuda.synchronize()

        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        val_preds.append(outputs)
        val_labels.append(targets)

        torch.cuda.synchronize()

        if args.distributed:
            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        running_loss += loss.item()

    val_preds = torch.cat(val_preds.cpu())
    val_labels = torch.cat(val_labels.cpu())

    torch.cuda.synchronize()

    if args.distributed and (args.dist_eval == True):
        dist.barrier()
        val_preds = tools.gather_tensor(val_preds)
        val_labels = tools.gather_tensor(val_labels)
    
    kappa_score, precision, recall, f1, accuracy, roc_auc, cm = calculate_metrics(val_preds, val_labels)
    avg_loss = running_loss /(len(data_loader)*args.world_size)
    
    return {"avg_loss":avg_loss, "kappa_score": kappa_score, "precision":precision, "recall":recall, "f1score": f1, "oacc":accuracy, "rocauc":roc_auc, "cm":cm}

@torch.no_grad()
def test_evaluate(data_loader, model, device, args):

    model.eval()
    test_preds = []
    test_labels = []
    for _, (samples, targets) in enumerate(data_loader, 0):
        samples = samples.to(dtype=torch.float32, device=device, non_blocking=True)
        targets = targets.to(dtype=torch.long, device=device, non_blocking=True)

        torch.cuda.synchronize()

        outputs = model(samples)
        
        test_preds.append(outputs.cpu())
        test_labels.append(targets.cpu())

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    torch.cuda.synchronize()

    if args.distributed and (args.dist_eval == True):
        dist.barrier()
        test_preds = tools.gather_tensor(test_preds)
        test_labels = tools.gather_tensor(test_labels)

    #remove background
    mask = (test_labels != 0)
    test_preds_noback = test_preds[mask]
    test_preds_noback = test_preds_noback-1
    test_labels = test_labels[mask]-1

    kappa_score, precision, recall, f1, accuracy, roc_auc, cm, per_class_accuracy, precision_class, recall_class, fscore_class, roc_per_class, support = calculate_test_metrics(test_preds_noback, test_labels)
    
    return torch.softmax(test_preds, dim=1) , {"kappa_score": kappa_score, "precision":precision, "recall":recall, "f1score": f1, "oacc":accuracy, "rocauc":roc_auc, "cm":cm, "per_class_accuracy":per_class_accuracy, "precision_class":precision_class, "recall_class":recall_class, "fscore_class":fscore_class, "roc_class":roc_per_class, "support":support}