import torch
import torch.nn as nn
from torchvision import transforms, utils
from torchvision import models as torch_models
from torch.autograd import Variable
import numpy as np
import scipy.misc
from PIL import Image
import random
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

import utils


# run.py initializes weights and save the weights as a file.
#  use_bn test 
# python .\run.py --use_bn --dataset imagenet --model resnet18
# python .\run.py --use_bn --dataset imagenet --model resnet50
# python .\run.py --dataset imagenet --model resnet18
# python .\run.py --dataset imagenet --model resnet50

# python .\run.py --dataset cifar10 --model resnet20
# python .\run.py --use_bn --dataset cifar10 --model resnet20

def chk_id_num(args, path):
    id_num = 0

    while(1):
        chk_file = os.path.isfile(path + args.model + '_' + args.dataset + '_' + str(id_num) + '.pth')
        if chk_file:
            id_num += 1
        else:
            return str(id_num)

parser = argparse.ArgumentParser()
# Architecture configs
parser.add_argument('-m', '--model', help='name of model in models', default='resnet34')
parser.add_argument('-d', '--dataset', help='name of target dataset', default='imagenet')
parser.add_argument('--use_bn', help='use batch normalization', action='store_true', default=False)

args = parser.parse_args()

weight_path = 'weights/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('ILSVRC2012_val_00043156.JPEG'))
plt.imshow(image)

# model = utils.get_model(None, args)
model = torch_models.resnet34(pretrained=True)
print(model)

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image

outputs = []
names = []
print(conv_layers)
idx = 0
for layer in conv_layers[0:]:
    image = layer(image)
    # print(image.shape)
    outputs.append(image)
    names.append(str(layer) + str(idx))

    fig = plt.figure(figsize=(30, 50))
    feature_map = image.squeeze(0)
    for i in range(20):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(feature_map[i].data.numpy())
        a.axis("off")
        a.set_title(str(i), fontsize=30)
    plt.savefig(str('feature_maps_' + str(idx) + '.jpg'), bbox_inches='tight')
    idx += 1
print(len(outputs))
#print feature_maps
# for feature_map in outputs:
#     print(feature_map.shape)



processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')