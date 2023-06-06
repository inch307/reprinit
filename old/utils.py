from platform import architecture
import torch, torchvision
import torch.nn as nn
import numpy as np
import os, pickle, random, math, datetime
from arch import ResNetArch, SimpleCNNArch
from protonet import ProtoNet
from dataset.dataset import Dataset
from repr_init import ReprInit
from dataset.prototypical_batch_sampler import PrototypicalBatchSampler

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

def init_sampler(args, labels, mode):
    if 'train' in mode:
        classes_per_it = args.classes_per_it_tr
        num_samples = args.num_support_tr + args.num_query_tr
    else:
        classes_per_it = args.classes_per_it_val
        num_samples = args.num_support_val + args.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=args.iterations)

def few_shot_data_loader(train_dataset, val_dataset, test_dataset, args):
    train_sampler = init_sampler(args, train_dataset.y, 'train')
    val_sampler = init_sampler(args, val_dataset.y, 'val')
    test_sampler = init_sampler(args, test_dataset.y, 'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_sampler=test_sampler,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader

def get_model(args, device):
    if args.arch == 'resnet18':
        arch = ResNetArch(arch=args.arch, model=args.model, dataset=args.dataset, out_dim=args.out_dim, device=device)
    elif args.arch == 'simplecnn':
        arch = SimpleCNNArch(arch=args.arch, model=args.model, dataset=args.dataset, out_dim=args.out_dim, device=device)
    elif args.arch == 'few_shot':
        arch = ProtoNet().to(device)
    return arch

def get_device(args):
    if args.disable_cuda:
        return torch.device('cpu')
    else:
        return torch.device('cuda:' + str(args.gpu_idx))

def weights_init_fan_in(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_fan_out(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_fan_in_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_fan_out_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

def get_dist_point_line(l, p):
    # l: (m, a, b)
    # p: (x1, y1)
    m, a, b = l
    x1, y1 = p
    return abs(m*x1 - y1 - m*a + b) / math.sqrt(m*m + 1)

def get_ab(R, m):
    sin = np.sqrt(1/(m**2 + 1))
    cos = -m * sin
    y1 = R*sin
    # y2 = -y1
    x1 = R*cos
    # x2 = -x1
    # y = -(1/m)(x-x1) + y1
    random_x = random.uniform(-x1, x1)
    y = -(1/m)*(random_x-x1) + y1

    return random_x, y


def get_cur_path():
    return os.getcwd()