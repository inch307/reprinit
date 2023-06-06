import torch
import random
import numpy as np
import argparse
import os

import utils
from reprinit import ReprInit

# run.py initializes weights and save the weights as a file.
#  use_bn test 
# python .\run.py --use_bn --dataset imagenet --model resnet18
# python .\run.py --use_bn --dataset imagenet --model resnet50
# python .\run.py --dataset imagenet --model resnet18
# python .\run.py --dataset imagenet --model resnet50

# python .\run.py --dataset cifar10 --model resnet20
# python .\run.py --use_bn --dataset cifar10 --model resnet20

def chk_id_num(path, args):
    id_num = 0

    while(1):
        chk_file = os.path.isfile(path + args.model + '_' + args.dataset + '_' + str(id_num) + '.pth')
        if chk_file:
            id_num += 1
        else:
            return str(id_num)

parser = argparse.ArgumentParser()
# Architecture configs
parser.add_argument('--base', default='../sdb1/')
parser.add_argument('-m', '--model', help='name of model in models', default='resnet20')
parser.add_argument('-d', '--dataset', help='name of target dataset', default='cifar10')
parser.add_argument('--use_bn', help='use batch normalization', action='store_true', default=False)
# ReprInit configs
# TODO: min max: beta, num_line, impact
# parser.add_argument('--fan_in', default=False, action='store_true', help='config for init (fan_in and fan_out). default is fan_out.')
# parser.add_argument('--beta', default=np.sqrt(2), help='hyperparameter beta')
parser.add_argument('--num_line', default=3, type=int, help='the number of lines to draw')
parser.add_argument('--impact', default=0.95, type=float, help='impact factor in Reprinit')
parser.add_argument('--seed', default=2023, type=int, help='seed for init')
parser.add_argument('--init_rate', default=0.7, type=float, help='ratio for reprinit and kaiming init')
parser.add_argument('--first_conv', action='store_true', default=False)
parser.add_argument('--first_conv_channel')
parser.add_argument('--no_zca', action='store_true', default=False)
# Other configs
parser.add_argument('-n', default=5, type=int, help='the number of weight files to generate')

args = parser.parse_args()

weight_path = 'weights/'

# init seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model, dataset_path = utils.get_model(args)
initializer = ReprInit(model, args)
weights = initializer.apply_initialization()

id_num = chk_id_num(weight_path, args)
torch.save(model.state_dict(), weight_path + args.model + '_' + args.dataset + '_' + id_num + '.pth')
# save_weights(weights, weight_path)