import argparse
import torch
import torchvision
import few_shot_train
from utils import *
from trainer import Trainer
from PIL import Image
import numpy as np
import datetime

parser = argparse.ArgumentParser(description='Pytorch "Repr intialization"')
parser.add_argument('-m', '--model', default='supervised', help='model', choices=['supervised', 'simclr', 'byol', 'proto', 'relation', 'siamese', 'matching'])
# parser.add_argument('--initial_linear_eval', default=False, action='store_true', help='linear evaluation on initialized model')
parser.add_argument('-d', '--dataset', default='cifar10', help='dataset-name', choices=['cifar10', 'stl10', 'imagenet', 'smallimagenet', 'omniglot', 'miniimagenet'])
parser.add_argument('-a', '--arch', default='resnet18', help='architecture of model')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of workers in dataloader')
parser.add_argument('-e', '--epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size of train')
parser.add_argument('-o', '--out_dim', default=128, type=int, help='out dim of resnset (only in constrastive learning)')
parser.add_argument('-t', '--temperature', default=0.07, type=float, help='temperature for simclr')
parser.add_argument('-i', '--iter_print', default=1000, type=int, help='iter to print')
parser.add_argument('-f', '--fan_in', default=False, action='store_true', help='default of init is fan out')
parser.add_argument('--n_runs', default=5, type=int, help='the number of runs')
parser.add_argument('--beta', default=np.sqrt(2), help='initialization parameter beta')
parser.add_argument('--repr_init', default=False, action='store_true', help='apply representation intiailization')
parser.add_argument('--impact', default=0.7, type=float, help='impact factor in repr init')
parser.add_argument('--adjusted_impact', default=False, action='store_true', help='adjusted impact in repr init')
parser.add_argument('--val_batch_size', default=256, type=int, help='batch size of validation')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training')
parser.add_argument('--disable_cuda', default=False, action='store_true', help='disable CUDA')
parser.add_argument('--gpu_idx', default=2, type=int, help='gpu index for gpu training')
parser.add_argument('--linear_eval', default=False, action='store_true', help='linear evaluation at every epoch')
parser.add_argument('--experiment', default='', help='experiment_name (logging model, weights, accuracy), if no experiment name, it does not log any data')
parser.add_argument('--iterations', default=100, type=int, help='number of episodes per epoch')

# N way
parser.add_argument('-cTr', '--classes_per_it_tr',
                    type=int,
                    help='number of random classes per episode for training, default=5',
                    default=5)

# K shoy
parser.add_argument('-nsTr', '--num_support_tr',
                    type=int,
                    help='number of samples per class to use as support for training, default=5',
                    default=1)

parser.add_argument('-nqTr', '--num_query_tr',
                    type=int,
                    help='num5er of samples per class to use as query for training, default=5',
                    default=15)

parser.add_argument('-cVa', '--classes_per_it_val',
                    type=int,
                    help='number of random classes per episode for validation, default=5',
                    default=5)

parser.add_argument('-nsVa', '--num_support_val',
                    type=int,
                    help='number of samples per class to use as support for validation, default=5',
                    default=1)

parser.add_argument('-nqVa', '--num_query_val',
                    type=int,
                    help='number of samples per class to use as query for validation, default=15',
                    default=15)
def main():
    args = parser.parse_args()
    device = get_device(args)
    log_path = logger_init(args)

    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for n in range(args.n_runs):
        train_dataset, val_dataset, test_dataset = get_dataset(args)
        # train_loader, val_loader, test_loader = get_data_loader(train_dataset, val_dataset, test_dataset, args)
        train_loader, val_loader, test_loader = few_shot_data_loader(train_dataset, val_dataset, test_dataset, args)
        net = get_model(args, device)
        # repr_init_rate = [0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        repr_init_rate = [0.8, 0.8, 0.8, 0.7]
        repr_init = ReprInit(net=net, args=args, device=device, repr_init_rate=repr_init_rate)
        repr_init.apply_initialization()
        torch.manual_seed(2022+n)
        torch.cuda.manual_seed(2022+n)

        # optimizer = torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), args.lr)

        # scheduler = None
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        # trainer = Trainer(train_loader=train_loader, val_loader=val_loader, net=net, optimizer=optimizer, scheduler=scheduler, device=device, n_runs=n,args=args, log_path=log_path)
                                
        # trainer.train()

        few_shot_train.few_shot_train(train_loader, val_loader, test_loader, net, optimizer, scheduler, args, device)

        ###################################
        if args.model == 'simclr':
            backup = args.model
            args.model = 'supervised'
            train_dataset, val_dataset = get_dataset(args)
            train_loader, val_loader = get_data_loader(train_dataset, val_dataset, args)
            net = get_model(args, device)
            for param in net.backbone.parameters():
                print('param')
                param.requires_grad = False
            dim_mlp = net.backbone.fc.in_features
            net.backbone.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
            if args.fan_in:
                net.apply(weights_init_fan_in)
            else:
                net.apply(weights_init_fan_out)
            # net.backbone.fc.weight.requires_grad = True
            # if net.backbone.fc.bias is not None:
            #     net.backbone.fc.bias.requires_grad = True

            args.epochs = 50

            optimizer = torch.optim.Adam(net.parameters(), args.lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
            temporal_name = TemporalName(model=args.model, train_loader=train_loader, val_loader=val_loader, net=net, optimizer=optimizer, scheduler=scheduler,
                                    device=device, epochs=args.epochs, batch_size=args.batch_size, temperature=args.temperature, iter_print=args.iter_print, n_runs=n, dataset=args.dataset, gpu_idx=args.gpu_idx, beta=args.beta, log_path=log_path)
                                
            temporal_name.train()
            args.model = backup


if __name__ == '__main__':
    main()
