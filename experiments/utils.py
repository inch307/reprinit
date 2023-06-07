from platform import architecture
import torch, torchvision
import torch.nn as nn
import numpy as np
import os, pickle, random, math, datetime
from dataset.dataset import Dataset
from ..models import resnet_cifar, resnet_imagenet

def mkdir(path):
    isExist = os.path.exists(path)
    if isExist:
        return
    else:
        os.mkdir(path)
        return

def chk_experiment_name(path):
    isExist = os.path.exists(path)
    if isExist:
        i = 1
        while(1):
            isExist = os.path.exists(path+'_'+str(i))
            if isExist:
                i += 1
            else:
                return path+'_'+str(i)+'/'
    else:
        return path+'/'

def logger_init(args):
    if args.experiment != '':
        cur_time = str(datetime.datetime.now())[0:16].replace('-','_').replace(' ','_')
        mkdir('./log/')
        log_path = chk_experiment_name('./log/'+args.experiment)
        mkdir(log_path)
        with open(log_path+'experiment_info.txt', 'w') as f:
            f.write('experiment date: '+cur_time+'\n\nargs:\n')
            for arg in vars(args):
                f.write(arg+': '+getattr(args, arg)+'\n')
            f.close()
        return log_path
    else:
        return ''


def get_dataset(args):
    data_path = get_cur_path() + '/data' + '/' + args.dataset
    if args.dataset == 'imagenet' or 'smallimagenet':
        data_path='../sdb1/ImageNet'
    dataset = Dataset(args=args, data_path=data_path)
    return dataset.get_dataset()

def get_data_loader(train_dataset, val_dataset, test_dataset, args):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader

def get_model(args, n_run):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.model == 'resnet20':
            model = resnet_cifar.resnet20(use_bn = args.use_bn)
        elif args.model == 'resnet32':
            model = resnet_cifar.resnet32(use_bn = args.use_bn)
        elif args.model == 'resnet44':
            model = resnet_cifar.resnet44(use_bn = args.use_bn)
        elif args.model == 'resnet56':
            model = resnet_cifar.resnet56(use_bn = args.use_bn)
        elif args.model == 'resnet110':
            model = resnet_cifar.resnet110(use_bn = args.use_bn)
        elif args.model == 'resnet1202':
            model = resnet_cifar.resnet1202(use_bn = args.use_bn)
        elif args.model == 'vit_b_16':
            model = torchvision.models.vit_b_16(image_size=32, path_size=4, num_classes=10)
        else:
            raise ValueError('Incorrect model name')
        
    elif args.dataset == 'imagenet':
        if args.model == 'resnet18':
            model = resnet_imagenet.resnet18(use_bn = args.use_bn)
        elif args.model == 'resnet34':
            model = resnet_imagenet.resnet34(use_bn = args.use_bn)
        elif args.model == 'resnet50':
            model = resnet_imagenet.resnet50(use_bn = args.use_bn)
        elif args.model == 'resnet101':
            model = resnet_imagenet.resnet101(use_bn = args.use_bn)
        elif args.model == 'resnet152':
            model = resnet_imagenet.resnet152(use_bn = args.use_bn)
        elif args.model == 'vit_b_16':
            model = torchvision.models.vit_b_16()
        else:
            raise ValueError('Incorrecnt model name')
    else:
        raise NotImplementedError('Not implemented dataset')
    
    torch.load(model)

    return model

def get_device(args):
    if args.disable_cuda:
        return torch.device('cpu')
    else:
        return torch.device('cuda:' + str(args.gpu_idx))

def get_cur_path():
    return os.getcwd()