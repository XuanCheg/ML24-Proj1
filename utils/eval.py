from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def eval_epoch(model, eval_dataset, opt, criterion=None):
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    misc = ['F1', 'Precision', 'Recall', 'Accuracy']
    

    metrics = {metric: 0 for metric in misc}
    loss_meters = []

    if opt.model in ['KNN', 'SVM']:
        for batch in eval_loader:
            inputs, targets = batch
            if opt.dset_name in ['ADNI_90_120_fMRI', 'FTD_90_200_fMRI', 'OCD_90_200_fMRI'] and opt.model in ['MLP', 'KNN', 'SVM']:
                inputs = inputs.flatten(start_dim=1)
            inputs, targets_np = inputs.numpy(), targets.numpy()
            preds_np = model(inputs)
            metrics['F1'] += f1_score(targets_np, preds_np, average='macro')
            metrics['Precision'] += precision_score(targets_np, preds_np, average='macro')
            metrics['Recall'] += recall_score(targets_np, preds_np, average='macro')
            metrics['Accuracy'] += accuracy_score(targets_np, preds_np)
            return metrics

    with torch.no_grad():
        for batch in eval_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)

            if opt.dset_name in ['ADNI_90_120_fMRI', 'FTD_90_200_fMRI', 'OCD_90_200_fMRI'] and opt.model == 'MLP':
                inputs = inputs.flatten(start_dim=1)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            if criterion is not None:
                loss = criterion(outputs, targets)
                loss_meters.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()

            metrics['F1'] += f1_score(targets_np, preds_np, average='macro')
            metrics['Precision'] += precision_score(targets_np, preds_np, average='macro')
            metrics['Recall'] += recall_score(targets_np, preds_np, average='macro')
            metrics['Accuracy'] += accuracy_score(targets_np, preds_np)

    num_batches = len(eval_loader)
    for metric in metrics:
        metrics[metric] /= num_batches

    return metrics