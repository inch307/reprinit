import logging
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
import utils

def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def topk_correct(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def val(val_loader, net, device, log):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct1 = 0
    # correct3 = 0
    correct5 = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            c1, c5 = topk_correct(output, target, (1, 5))
            correct1 += c1
            # correct3 += c3
            correct5 += c5
    acc1 = correct1 / len(val_loader.dataset) * 100
    # acc3 = correct3 / len(val_loader.dataset) * 100
    acc5 = correct5 / len(val_loader.dataset) * 100

    log['val_acc1'].append(acc1)
    # log['val_acc3'].append(acc3)
    log['val_acc5'].append(acc5)

    return acc1, acc5
    

def train(net, train_loader, val_loader, optimizer, scheduler, device, epochs, log):
    net.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # if n_iter % args.iter_print == 0:
            #     correct1, correct3, correct5 = topk_correct(output, target, (1, 3, 5))
            #     acc1_i = correct1 / target.size(0) * 100
            #     acc3_i = correct3 / target.size(0) * 100
            #     acc5_i = correct5 / target.size(0) * 100
            #     print(f'batch acc1: {acc1_i}, acc3: {acc3_i}, acc5: {acc5_i}, loss: {loss}')
            #     # self.log['train_acc1'].append(acc1)
            #     # self.log['train_acc3'].append(acc3)
            #     # self.log['train_acc5'].append(acc5)
            # n_iter += 1

            correct1, correct5 = topk_correct(output, target, (1, 3, 5))
            acc1 = correct1 / target.size(0) * 100
            # acc3 = correct3 / target.size(0) * 100
            acc5 = correct5 / target.size(0) * 100
            log['train_acc1'].append(acc1)
            # log['train_acc3'].append(acc3)
            log['train_acc5'].append(acc5)
            # print(f'At epoch {epoch} train acc1: {acc1}, acc5: {acc5}')
        val_acc1, val_acc5 = val(val_loader, net, device, log)
        print(f'Validation at epoch {epoch}, acc1: {val_acc1}, acc5: {val_acc5}')