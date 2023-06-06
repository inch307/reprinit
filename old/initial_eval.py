import argparse
import torch
import torchvision
from utils import *
from trainer import TemporalName
from PIL import Image
import numpy as np
import datetime

parser = argparse.ArgumentParser(description='Pytorch "temporal name"')
parser.add_argument('-m', '--model', default='supervised', help='model', choices=['supervised', 'simclr', 'byol'])
parser.add_argument('--initial_linear_eval', default=False, action='store_true', help='linear evaluation on initialized model')
parser.add_argument('-d', '--dataset', default='cifar10', help='dataset-name', choices=['cifar10', 'stl10', 'imagenet'])
parser.add_argument('-a', '--arch', default='resnet18', help='architecture of model')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of workers in dataloader')
parser.add_argument('-e', '--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size of train')
parser.add_argument('-o', '--out_dim', default=128, type=int, help='out dim of resnset (only in constrastive learning)')
parser.add_argument('-t', '--temperature', default=0.07, type=float, help='temperature for simclr')
parser.add_argument('-i', '--iter_print', default=1000, type=int, help='iter to print')
parser.add_argument('-f', '--fan_in', default=False, action='store_true', help='default of init is fan out')
parser.add_argument('--n_runs', default=5, type=int, help='the number of runs')
parser.add_argument('--beta', default=np.sqrt(2), help='initialization parameter beta')
parser.add_argument('--repr_initialization', default=False, action='store_true', help='apply representation intiailization')
parser.add_argument('--num_cz_sampling', default=1000000, type=int, help='the number of sampling when repr init')
parser.add_argument('--impact', default=0.7, type=float, help='impact factor in repr init')
parser.add_argument('--val_batch_size', default=256, type=int, help='batch size of validation')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training')
parser.add_argument('--disable_cuda', default=False, action='store_true', help='disable CUDA')
parser.add_argument('--gpu_idx', default=0, type=int, help='gpu index for gpu training')
parser.add_argument('--linear_eval', default=False, action='store_true', help='fine tuning')

def main():
    args = parser.parse_args()
    print('arg')

    device = get_device(args)

    cur_time = str(datetime.datetime.now())[5:13].replace('-','_').replace(' ','_')
    log_path = './log/'+cur_time+'_m'+args.model+'_'+args.dataset+'_d'+str(args.gpu_idx)

    for n in range(args.n_runs):
        train_dataset, val_dataset = get_dataset(args)
        print('dataset')

        train_loader, val_loader = get_data_loader(train_dataset, val_dataset, args)
        print('train loader')

        net = get_model(args, device)
        print('net')

        apply_initialization(net, device, args)
        for param in net.backbone.parameters():
            print('param')
            param.requires_grad = False
        net.backbone.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
        net.apply(weights_init_fan_in)

        # optimizer = torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), args.lr)

        # scheduler = None
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        temporal_name = TemporalName(model=args.model, train_loader=train_loader, val_loader=val_loader, net=net, optimizer=optimizer, scheduler=scheduler,
                                    device=device, epochs=args.epochs, batch_size=args.batch_size, temperature=args.temperature, iter_print=args.iter_print, n_runs=n, dataset=args.dataset, gpu_idx=args.gpu_idx, beta=args.beta, log_path=log_path)
                                
        temporal_name.train()

if __name__ == '__main__':
    main()
