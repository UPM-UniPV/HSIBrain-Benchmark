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
    val_preds_softmax = torch.softmax(preds, dim=1).cpu().numpy()
    val_preds_argmax = np.argmax(val_preds_softmax, axis=1)
    #al_preds_argmax = torch.argmax(preds, dim=1).cpu().numpy()
    val_labels = labels.cpu().numpy()

    correct = (val_preds_argmax == val_labels).sum().item()

    precision = precision_score(val_labels, val_preds_argmax, average='micro', zero_division=0)
    recall = recall_score(val_labels, val_preds_argmax, average='micro', zero_division=0)
    f1 = f1_score(val_labels, val_preds_argmax, average='micro', zero_division=0)

    cm = confusion_matrix(val_labels, val_preds_argmax)

    accuracy = correct / len(val_labels)

    roc_auc = roc_auc_score(val_labels, val_preds_softmax, multi_class='ovo', average='weighted', labels=[0,1,2,3])

    kappa_score = cohen_kappa_score(val_labels, val_preds_argmax)

    return kappa_score, precision, recall, f1, accuracy, roc_auc, cm

def calculate_test_metrics(preds, labels):
    test_preds_softmax = torch.softmax(preds,dim=1).cpu().numpy()
    test_preds_argmax = np.argmax(test_preds_softmax, axis=1)
    #test_preds_argmax = torch.argmax(preds, dim=1).cpu().numpy()
    test_labels = labels.cpu().numpy()

    correct = (test_preds_argmax == test_labels).sum().item()
    cm = confusion_matrix(test_labels, test_preds_argmax)

    #overall
    accuracy = correct / len(test_labels)
    precision = precision_score(test_labels, test_preds_argmax, average='micro', zero_division=0)
    recall = recall_score(test_labels, test_preds_argmax, average='micro', zero_division=0)
    f1 = f1_score(test_labels, test_preds_argmax, average='micro', zero_division=0)

    #roc per class + overall
    roc_per_class = []
    roc_auc = 0.0
    if(len(np.unique(test_labels)) == 4): #if all classes are present
        roc_per_class = roc_auc_score(test_labels, test_preds_softmax, multi_class='ovr', average='weighted', labels=[0,1,2,3])
        roc_auc = roc_auc_score(test_labels, test_preds_softmax, multi_class='ovo', average='weighted', labels=[0,1,2,3])

    #per class
    precision_class, recall_class, fscore_class, support = precision_recall_fscore_support(test_labels, test_preds_argmax, beta=1.0, average=None, zero_division=0)
    per_class_accuracy = np.where(cm.sum(axis=1) == 0, 0, cm.diagonal() / cm.sum(axis=1))

    kappa_score = cohen_kappa_score(test_labels, test_preds_argmax)

    return kappa_score, precision, recall, f1, accuracy, roc_auc, cm, per_class_accuracy, precision_class, recall_class, fscore_class, roc_per_class, support

def train_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, criterion, args):
    
    model.train(True)
    running_loss = 0.0
    for _, (samples, targets) in enumerate(data_loader, 0):        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()

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
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        val_preds.append(outputs)
        val_labels.append(targets)

        torch.cuda.synchronize()

        if args.distributed:
            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        running_loss += loss.item()

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

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
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        
        test_preds.append(outputs)
        test_labels.append(targets)

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