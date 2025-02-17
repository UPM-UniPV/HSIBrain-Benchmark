"""
Train and eval functions
"""
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)

import utils.tools as tools

import time


def calc_auc(true_labels, pred_labels):
    """ AUC """
    auc_class = np.array([-1., -1., -1., -1.])
    auc_wavg = -1.
    roc_auc_list, weights = [], []
    n_samples = len(true_labels)

    healthy_gt = np.sum(true_labels == 0) > 0
    tumor_gt = np.sum(true_labels == 1) > 0
    vessel_gt = np.sum(true_labels == 2) > 0
    outer_gt = np.sum(true_labels == 3) > 0

    active_vect = np.where(
        np.array([healthy_gt, tumor_gt, vessel_gt, outer_gt]))[0]

    if len(active_vect) > 1:
        for lab in active_vect:
            y_binary = (true_labels == lab).astype(int)
            y_prob_l = pred_labels[:, lab]

            # Calculate ROC AUC for the current class
            try:
                roc_auc = roc_auc_score(y_binary, y_prob_l)
                roc_auc_list.append(roc_auc)

                # Calculate the weight for the current class
                weight = np.sum(y_binary) / n_samples
                weights.append(weight)
            except ValueError:
                print(f"ROC AUC couldn't be calculated for class {lab}")
                roc_auc_list.append(-1)
                weights.append(0)

        auc_c = np.array(roc_auc_list)
        auc_wavg = np.sum(auc_c * np.array(weights))
        auc_class[active_vect] = auc_c

    return auc_class, auc_wavg


def calculate_metrics(preds, labels):
    val_preds_softmax = torch.softmax(preds, dim=1).numpy()
    val_preds_argmax = np.argmax(val_preds_softmax, axis=1)
    val_labels = labels.numpy()

    correct = (val_preds_argmax == val_labels).sum().item()

    precision = precision_score(val_labels, val_preds_argmax,
                                average='weighted', zero_division=0, labels=[0, 1, 2, 3])
    recall = recall_score(val_labels, val_preds_argmax,
                          average='weighted', zero_division=0, labels=[0, 1, 2, 3])
    f1 = f1_score(val_labels, val_preds_argmax, average='weighted',
                  zero_division=0, labels=[0, 1, 2, 3])

    cm = confusion_matrix(val_labels, val_preds_argmax, labels=[0, 1, 2, 3])

    accuracy = correct / len(val_labels)

    auc_class, auc_wavg = calc_auc(val_labels, val_preds_softmax)

    kappa_score = cohen_kappa_score(
        val_labels, val_preds_argmax, labels=[0, 1, 2, 3])
    if len(set(val_labels)) == 1:
        kappa_score = -1

    return kappa_score, precision, recall, f1, accuracy, auc_wavg, auc_class, cm


def calculate_test_metrics(preds, labels):
    test_preds_softmax = torch.softmax(preds, dim=1).numpy()
    test_preds_argmax = np.argmax(test_preds_softmax, axis=1)
    test_labels = labels.numpy()

    correct = (test_preds_argmax == test_labels).sum().item()
    cm = confusion_matrix(test_labels, test_preds_argmax, labels=[0, 1, 2, 3])

    # overall
    accuracy = correct / len(test_labels)
    precision = precision_score(test_labels, test_preds_argmax,
                                average='weighted', zero_division=0, labels=[0, 1, 2, 3])
    recall = recall_score(test_labels, test_preds_argmax,
                          average='weighted', zero_division=0, labels=[0, 1, 2, 3])
    f1 = f1_score(test_labels, test_preds_argmax,
                  average='weighted', zero_division=0, labels=[0, 1, 2, 3])

    auc_class, auc_wavg = calc_auc(test_labels, test_preds_softmax)

    # per class
    precision_class, recall_class, fscore_class, support = precision_recall_fscore_support(
        test_labels, test_preds_argmax, beta=1.0, average=None, zero_division=0, labels=[0, 1, 2, 3])

    precision_class = np.where(support == 0, -1, precision_class)
    recall_class = np.where(support == 0, -1, recall_class)
    fscore_class = np.where(support == 0, -1, fscore_class)

    per_class_accuracy = np.where(
        support == 0, -1, cm.diagonal() / np.maximum(cm.sum(axis=1), 1))

    kappa_score = cohen_kappa_score(
        test_labels, test_preds_argmax, labels=[0, 1, 2, 3])
    if len(set(test_labels)) == 1:
        kappa_score = -1

    return kappa_score, precision, recall, f1, accuracy, auc_wavg, cm, per_class_accuracy, precision_class, recall_class, fscore_class, auc_class, support


def train_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, criterion, args):

    model.train(True)
    running_loss = 0.0
    for _, (samples, targets) in enumerate(data_loader, 0):
        samples = samples.to(dtype=torch.float32,
                             device=device, non_blocking=True)
        targets = targets.to(
            dtype=torch.long, device=device, non_blocking=True)

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

    # world_size = 1 if not distributed
    avg_loss = running_loss / (len(data_loader) * args.world_size)
    return {"avg_loss": avg_loss}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args):

    model.eval()
    val_preds = []
    val_labels = []
    running_loss = 0.0
    for _, (samples, targets) in enumerate(data_loader, 0):
        samples = samples.to(dtype=torch.float32,
                             device=device, non_blocking=True)
        targets = targets.to(
            dtype=torch.long, device=device, non_blocking=True)

        torch.cuda.synchronize()

        outputs = model(samples)
        loss = criterion(outputs, targets)

        val_preds.append(outputs.cpu())
        val_labels.append(targets.cpu())

        torch.cuda.synchronize()

        if args.distributed:
            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        running_loss += loss.item()

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    torch.cuda.synchronize()

    if args.distributed and (args.dist_eval is True):
        dist.barrier()
        val_preds = tools.gather_tensor(val_preds)
        val_labels = tools.gather_tensor(val_labels)

    kappa_score, precision, recall, f1, accuracy, roc_auc, roc_auc_class, cm = calculate_metrics(
        val_preds, val_labels)
    avg_loss = running_loss / (len(data_loader) * args.world_size)

    return {"avg_loss": avg_loss, "kappa_score": kappa_score, "precision": precision, "recall": recall, "f1score": f1, "oacc": accuracy,
            "rocauc": roc_auc, "rocHT": roc_auc_class[0], "rocTT": roc_auc_class[1], "rocBT": roc_auc_class[2], "rocDM": roc_auc_class[3], "cm": cm}


@torch.no_grad()
def test_evaluate(data_loader, model, device, args):

    model.eval()
    test_preds = []
    test_labels = []

    start_time = time.time()

    for _, (samples, targets) in enumerate(data_loader, 0):
        samples = samples.to(dtype=torch.float32,
                             device=device, non_blocking=True)
        targets = targets.to(
            dtype=torch.long, device=device, non_blocking=True)

        torch.cuda.synchronize()

        outputs = model(samples)

        test_preds.append(outputs.cpu())
        test_labels.append(targets.cpu())

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    torch.cuda.synchronize()

    if args.distributed and (args.dist_eval is True):
        dist.barrier()
        test_preds = tools.gather_tensor(test_preds)
        test_labels = tools.gather_tensor(test_labels)

    end_time = time.time()
    inference_time = end_time - start_time

    # remove background
    mask = (test_labels != 0)
    test_preds_noback = test_preds[mask]
    test_preds_noback = test_preds_noback - 1
    test_labels = test_labels[mask] - 1

    kappa_score, precision, recall, f1, accuracy, roc_auc, cm, per_class_accuracy, precision_class, recall_class, fscore_class, roc_per_class, support = calculate_test_metrics(
        test_preds_noback, test_labels)

    return torch.softmax(test_preds, dim=1), {"kappa_score": kappa_score, "precision": precision, "recall": recall, "f1score": f1, "oacc": accuracy, "rocauc": roc_auc, "cm": cm, "per_class_accuracy": per_class_accuracy,
                                              "precision_class": precision_class, "recall_class": recall_class, "fscore_class": fscore_class, "roc_class": roc_per_class, "support": support, "inference_time": inference_time}
