from platform import architecture
import torch, torchvision
import torch.nn as nn
import numpy as np
import os, pickle, random, math, datetime

import models.resnet_cifar
import models.resnet_imagenet



def get_model(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.dataset == 'cifar10':
            dataset_path = args.base + 'cifar10/'
        else:
            dataset_path = args.base + 'cifar100/'
        if args.model == 'resnet20':
            model = models.resnet_cifar.resnet20(use_bn = args.use_bn)
        elif args.model == 'resnet32':
            model = models.resnet_cifar.resnet32(use_bn = args.use_bn)
        elif args.model == 'resnet44':
            model = models.resnet_cifar.resnet44(use_bn = args.use_bn)
        elif args.model == 'resnet56':
            model = models.resnet_cifar.resnet56(use_bn = args.use_bn)
        elif args.model == 'resnet110':
            model = models.resnet_cifar.resnet110(use_bn = args.use_bn)
        elif args.model == 'resnet1202':
            model = models.resnet_cifar.resnet1202(use_bn = args.use_bn)
        elif args.model == 'vit_b_16':
            model = torchvision.models.vit_b_16(image_size=32, path_size=4, num_classes=10)
        else:
            raise ValueError('Incorrect model name')
        
    elif args.dataset == 'imagenet':
        dataset_path = args.base + 'ImageNet/'
        if args.model == 'resnet18':
            model = models.resnet_imagenet.resnet18(use_bn = args.use_bn)
        elif args.model == 'resnet34':
            model = models.resnet_imagenet.resnet34(use_bn = args.use_bn)
        elif args.model == 'resnet50':
            model = models.resnet_imagenet.resnet50(use_bn = args.use_bn)
        elif args.model == 'resnet101':
            model = models.resnet_imagenet.resnet101(use_bn = args.use_bn)
        elif args.model == 'resnet152':
            model = models.resnet_imagenet.resnet152(use_bn = args.use_bn)
        elif args.model == 'vit_b_16':
            model = torchvision.models.vit_b_16()
        else:
            raise ValueError('Incorrecnt model name')
    else:
        raise NotImplementedError('Not implemented dataset')
    
    return model, dataset_path

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

def weights_init_batchnorm(m):
    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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