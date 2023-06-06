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

torch.manual_seed(0)

class Trainer():
    def __init__(self, *args, **kwargs):
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.device = kwargs['device']
        self.n_runs = kwargs['n_runs']
        self.log_path = kwargs['log_path']
        self.args = kwargs['args']
        self.log = {'model':self.args.model, 'arch':self.net.arch, 'dataset': self.args.dataset, 'beta':float(self.args.beta), 'n_runs': self.n_runs, 'epochs':self.args.epochs, 'batch_size':self.args.batch_size, 'train_acc1': [], 'train_acc3': [], 'train_acc5': [], 'val_acc1': [], 'val_acc3':[], 'val_acc5':[]}
        self.n_iter = 0
        if self.n_runs == -1:
            self.epo = kwargs['epo']

        self.net = self.net.to(self.device)

    def topk_accuracy(self, output, target, topk=(1,)):
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

    def topk_correct(self, output, target, topk=(1,)):
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

    def info_nce_loss(self, features):

        # generate label number [0, 1, 2 ,..., 15, 0, 1, 2, ..., 15]
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        # marking positive sample [1, 0, ..., 1 (16-th element), ..., 0]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives shape: [32, 1]
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives shape: [32,30]
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # [32, 31]
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / float(self.args.temperature)
        return logits, labels

    def val_supervised(self):
        self.net.backbone.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct1 = 0
        correct3 = 0
        correct5 = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += criterion(output, target).item()
                c1, c3, c5 = self.topk_correct(output, target, (1, 3, 5))
                correct1 += c1
                correct3 += c3
                correct5 += c5
        acc1 = correct1 / len(self.val_loader.dataset) * 100
        acc3 = correct3 / len(self.val_loader.dataset) * 100
        acc5 = correct5 / len(self.val_loader.dataset) * 100

        self.log['val_acc1'].append(acc1)
        self.log['val_acc3'].append(acc3)
        self.log['val_acc5'].append(acc5)

        print(f'validation, acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')

    def train_supervised(self):
        self.net.backbone.train()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for _batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if self.n_iter % self.args.iter_print == 0:
                correct1, correct3, correct5 = self.topk_correct(output, target, (1, 3, 5))
                acc1_i = correct1 / target.size(0) * 100
                acc3_i = correct3 / target.size(0) * 100
                acc5_i = correct5 / target.size(0) * 100
                print(f'batch acc1: {acc1_i}, acc3: {acc3_i}, acc5: {acc5_i}, loss: {loss}')
                # self.log['train_acc1'].append(acc1)
                # self.log['train_acc3'].append(acc3)
                # self.log['train_acc5'].append(acc5)
            self.n_iter += 1

        correct1, correct3, correct5 = self.topk_correct(output, target, (1, 3, 5))
        acc1 = correct1 / target.size(0) * 100
        acc3 = correct3 / target.size(0) * 100
        acc5 = correct5 / target.size(0) * 100
        self.log['train_acc1'].append(acc1)
        self.log['train_acc3'].append(acc3)
        self.log['train_acc5'].append(acc5)
        print(f'batch acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')

    def train_simclr(self):
        self.net.backbone.train()
        criterion = nn.CrossEntropyLoss()
        for data, _ in tqdm(self.train_loader):
            data = torch.cat(data, dim=0)
            data = data.to(self.device)

            features = self.net(data)
            logits, labels = self.info_nce_loss(features)
            loss = criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.n_iter % self.args.iter_print == 0:
                correct1, correct3, correct5 = self.topk_correct(logits, labels, (1, 3, 5))
                acc1_i = correct1 / len(labels) * 100
                acc3_i = correct3 / len(labels) * 100
                acc5_i = correct5 / len(labels) * 100
                print(f'batch acc1: {acc1_i}, acc3: {acc3_i}, acc5: {acc5_i}, loss: {loss}')
            self.n_iter += 1
            
        correct1, correct3, correct5 = self.topk_correct(logits, labels, (1, 3, 5))
        acc1 = correct1 / len(labels) * 100
        acc3 = correct3 / len(labels) * 100
        acc5 = correct5 / len(labels) * 100
        self.log['train_acc1'].append(acc1)
        self.log['train_acc3'].append(acc3)
        self.log['train_acc5'].append(acc5)
        print(f'batch acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')
            
    def val_simclr(self):
        if self.args.dataset == 'imagenet' or 'smallimagenet':
            num_classes = 1000
        else:
            num_classes = 10
        for name, param in self.net.backbone.named_parameters():
            param.requires_grad = False
            
        dim_mlp = self.net.backbone.fc[0].in_features
        backup_fc = self.net.backbone.fc
        self.net.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)).to(self.device)

        class TmpArgs():
            def __init__(self, *args, **kwargs):
                self.model = 'supervised'
                self.dataset = kwargs['dataset']
                self.batch_size = kwargs['batch_size']
                self.val_batch_size = kwargs['val_batch_size']
                self.workers = kwargs['workers']
        tmp_args = TmpArgs(dataset=self.args.dataset, batch_size=self.args.batch_size, val_batch_size=self.args.val_batch_size, workers=self.args.workers)
        train_dataset, val_dataset = utils.get_dataset(tmp_args)
        train_loader, val_loader = utils.get_data_loader(train_dataset, val_dataset, tmp_args)
        ##################### train
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        for epo in range(30):
            self.net.backbone.train()
            criterion = nn.CrossEntropyLoss()
            losses = []
            for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.net(data)
                loss = criterion(output, target)
                loss.backward()

                optimizer.step()
                
                losses.append(loss.item())

            correct1, correct3, correct5 = self.topk_correct(output, target, (1, 3, 5))
            acc1 = correct1 / target.size(0) * 100
            acc3 = correct3 / target.size(0) * 100
            acc5 = correct5 / target.size(0) * 100
            self.log['train_acc1'].append(acc1)
            self.log['train_acc3'].append(acc3)
            self.log['train_acc5'].append(acc5)
            print(f'linear eval train at epoch: {epo}, acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')
        ###################### train end

            self.net.backbone.eval()
            criterion = nn.CrossEntropyLoss()
            test_loss = 0
            correct1 = 0
            correct3 = 0
            correct5 = 0

            with torch.no_grad():
                for data, target in tqdm(val_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.net(data)
                    test_loss += criterion(output, target).item()
                    c1, c3, c5 = self.topk_correct(output, target, (1, 3, 5))
                    correct1 += c1
                    correct3 += c3
                    correct5 += c5
            acc1 = correct1 / len(val_loader.dataset) * 100
            acc3 = correct3 / len(val_loader.dataset) * 100
            acc5 = correct5 / len(val_loader.dataset) * 100

            self.log['val_acc1'].append(acc1)
            self.log['val_acc3'].append(acc3)
            self.log['val_acc5'].append(acc5)

            print(f'linear eval validation at epoch: {epo}, acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')
        self.net.backbone.fc = backup_fc
        for param in self.net.backbone.parameters():
            param.requires_grad = True

    def train_byol(self):
        pass

    def train(self):
        # if not os.path.isdir(self.log_path):
        #     os.mkdir(self.log_path)
        if self.args.model == 'supervised':
            for epoch in range(self.args.epochs):
                print(f'epoch is {epoch}')
                self.train_supervised()
                # cur_time = str(datetime.datetime.now())[5:13].replace('-','_').replace(' ','_')
                # if self.n_runs >= 0:
                #     if epoch % 10 == 0:
                #         torch.save(self.net.backbone.state_dict(), self.log_path+'/'+cur_time+'_'+self.net.arch+'_m'+self.model+'_'+self.dataset+'_d'+str(self.gpu_idx)+'_n'+str(self.n_runs)+'_e'+str(epoch)+'.pt')
                # else:
                #     print(f'original epoch is {self.epo}')
                self.val_supervised()
            # if self.n_runs >= 0:
            #     with open(self.log_path+'/'+cur_time+'_'+self.net.arch+'_m'+self.model+'_d'+str(self.gpu_idx)+'_n'+str(self.n_runs)+'_log.pk', 'wb') as f:
            #         pickle.dump(self.log, f)
            #         print(self.log)
            # else:
            #     with open(self.log_path+'/'+cur_time+'_'+self.net.arch+'_m'+self.model+'_d'+str(self.gpu_idx)+'_e'+str(self.epo)+'_log.pk', 'wb') as f:
            #         pickle.dump(self.log, f)
            #         print(self.log)
        elif self.args.model == 'simclr':
            for epoch in range(self.args.epochs):
                print(f'epoch is {epoch}')
                self.train_simclr()
                # cur_time = str(datetime.datetime.now())[5:13].replace('-','_').replace(' ','_')
                # if epoch % 10 == 0:
                #     torch.save(self.net.backbone.state_dict(), self.log_path+'/'+cur_time+'_'+self.net.arch+'_m'+self.model+'_'+self.dataset+'_d'+str(self.gpu_idx)+'_n'+str(self.n_runs)+'_e'+str(epoch)+'.pt')
                if epoch % 5 == 0:
                    self.val_simclr()
            # with open(self.log_path+'/'+cur_time+'_'+self.net.arch+'_m'+self.model+'_d'+str(self.gpu_idx)+'_log.pk', 'wb') as f:
            #     pickle.dump(self.log, f)
            #     print(self.log)
        elif self.args.model == 'byol':
            for epoch in range(self.args.epochs):
                self.train_byol()