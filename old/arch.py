from torchvision import models
import torch.nn as nn
import resnet55

class ResNetArch(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNetArch, self).__init__()
        self.arch = kwargs['arch']
        self.model = kwargs['model']
        self.dataset = kwargs['dataset']
        self.out_dim = kwargs['out_dim']
        self.device = kwargs['device']
        self.backbone = self.get_backbone().to(self.device)

        
        if self.model == 'simclr' or self.model == 'byol':
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
            # self.backbone.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def get_backbone(self):
        out_dim = self.get_out_dim()      

        if self.arch == 'resnet18':
            return models.resnet18(pretrained=False, num_classes=out_dim)
        elif self.arch == 'resnet50':
            return models.resnet50(pretrained=False, num_classes=out_dim)
        elif self.arch == 'resnet18_55':
            print('resnet18_55')
            return resnet55.resnet18(pretrained=False, num_classes=out_dim)
        elif self.arch == 'resnet50_55':
            print('resnet50_55')
            return resnet55.resnet50(pretrained=False, num_classes=out_dim)
        else:
            return models.resnet18(pretrained=False, num_classes=out_dim)

    def get_out_dim(self):
        if self.model == 'supervised':
            if self.dataset == 'cifar10':
                return 10
            elif self.dataset == 'stl10':
                return 10
            elif self.dataset == 'imagenet' or 'smallimagenet': 
                return 1000
        
        else:
            return self.out_dim

    def forward(self, x):
        return self.backbone(x)

class SimpleCNNArch(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNetArch, self).__init__()
        self.arch = kwargs['arch']
        self.model = kwargs['model']
        self.dataset = kwargs['dataset']
        self.out_dim = kwargs['out_dim']
        self.device = kwargs['device']
        self.backbone = self.get_backbone().to(self.device)

        
        if self.model == 'simclr' or self.model == 'byol':
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
            # self.backbone.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def get_backbone(self):
        out_dim = self.get_out_dim()      

        if self.arch == 'resnet18':
            return models.resnet18(pretrained=False, num_classes=out_dim)
        elif self.arch == 'resnet50':
            return models.resnet50(pretrained=False, num_classes=out_dim)
        elif self.arch == 'resnet18_55':
            print('resnet18_55')
            return resnet55.resnet18(pretrained=False, num_classes=out_dim)
        elif self.arch == 'resnet50_55':
            print('resnet50_55')
            return resnet55.resnet50(pretrained=False, num_classes=out_dim)
        else:
            return models.resnet18(pretrained=False, num_classes=out_dim)

    def get_out_dim(self):
        if self.model == 'supervised':
            if self.dataset == 'cifar10':
                return 10
            elif self.dataset == 'stl10':
                return 10
            elif self.dataset == 'imagenet' or 'smallimagenet': 
                return 1000
        
        else:
            return self.out_dim

    def forward(self, x):
        return self.backbone(x)
