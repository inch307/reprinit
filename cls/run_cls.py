import argparse
import torch
import torchvision
import pytorch_warmup as warmup

# import utils
from PIL import Image
import numpy as np
import datetime
import train_cls
from utils import *

# python run_cls.py --dataset imagenet --model resnet18 --epochs 200 -b 256 --weight_path resnet18_imagenet_0.pth --gpu_idx 0 --use_bn
# python run_cls.py --dataset imagenet --model resnet18 --epochs 200 -b 256 --gpu_idx 1 --use_bn
# python run_cls.py --dataset imagenet --model resnet18 --epochs 200 -b 256 --weight_path resnet18_imagenet_1.pth --gpu_idx 2 --use_bn

parser = argparse.ArgumentParser(description='Pytorch "Repr intialization"')
parser.add_argument('-d', '--dataset', default='cifar10', help='dataset-name', choices=['cifar10', 'stl10', 'imagenet', 'smallimagenet', 'omniglot', 'miniimagenet'])
parser.add_argument('-a', '--model', default='resnet18', help='architecture of model')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of workers in dataloader')
parser.add_argument('-e', '--epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size of train')
parser.add_argument('--use_bn', default=False, action='store_true')
parser.add_argument('--weight_path', default=None)
# parser.add_argument('-i', '--iter_print', default=1000, type=int, help='iter to print')
parser.add_argument('--n_runs', default=5, type=int, help='the number of runs')
parser.add_argument('--val_batch_size', default=256, type=int, help='batch size of validation')
parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')
parser.add_argument('--disable_cuda', default=False, action='store_true', help='disable CUDA')
parser.add_argument('--gpu_idx', default=0, type=int, help='gpu index for gpu training')

def main():
    args = parser.parse_args()
    device = get_device(args)
    log, log_path = logger_init(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for n in range(args.n_runs):
        train_dataset, val_dataset, test_dataset = get_dataset(args)
        # train_loader, val_loader, test_loader = get_data_loader(train_dataset, val_dataset, test_dataset, args)
        train_loader, val_loader, test_loader = get_data_loader(train_dataset, val_dataset, test_dataset, args)
        net = get_model(args)
        net.to(device)

        # optimizer = torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.weight_decay)

        # scheduler = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        warmup_scheduler = None
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        train_cls.train(net, train_loader, val_loader, test_loader, optimizer, scheduler, warmup_scheduler, device, args.epochs, log, log_path)

if __name__ == '__main__':
    main()
