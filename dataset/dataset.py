from torchvision import datasets, transforms
import torch
import os
from PIL import Image
import scipy.io as sio
from .data_aug import ContrastiveLearningDataset
from .view_generator import ContrastiveLearningViewGenerator
from .imagenet_dataset import ImagenetTrainDataset, ImagenetTestDataset
from .omniglot_dataset import OmniglotDataset

class Dataset:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = self.args.model
        self.dataset = self.args.dataset
        self.data_path = kwargs['data_path']

    def get_dataset(self):
        if self.model == 'supervised':
            if self.dataset == 'stl10':
                return self.supervised_stl10_dataset()
            elif self.dataset == 'imagenet':
                return self.supervised_imagenet_dataset()
            else:
                return self.supervised_cifar10_dataset()

        elif self.model == 'simclr' or self.model == 'byol':
            if self.dataset == 'stl10':
                return self.self_supervised_stl10_dataset()
            elif self.dataset == 'imagenet':
                return self.self_supervised_imagenet_dataset()
            else:
                return self.self_supervised_cifar10_dataset()

        elif self.model == 'proto' or self.model == 'relation' or self.model == 'siamese' or self.model == 'matching':
            if self.dataset == 'omniglot': # few shot (ref. relation network)
                return self.omniglot_dataset()
            elif self.dataset == 'miniimagenet': # few shot (ref. relation network)
                return self.miniimagenet_dataset() 
            elif self.dataset == 'cub': # zero shot cal-tech UCSD birds (ref. relation network)
                return self.cub_dataset()
            elif self.dataset == 'awa1': # zero shot animal with attributes  (ref. relation network)
                return self.awa1_dataset()
            elif self.dataset == 'awa2': # zero shot animal with attributes (ref. relation network)
                return self.awa2_dataset()

    def supervised_stl10_dataset(self):
        train_dataset = datasets.STL10(self.data_path, download=True, split='train',
                                        transform=transforms.Compose(
                                            [
                                                transforms.RandomCrop(96, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                            ]
                                        )
        )
        val_dataset = datasets.STL10(self.data_path, download=True, split='test',
                                        transform=transforms.Compose(
                                            [
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                            ]
                                        )
        )

        return train_dataset, val_dataset

    def supervised_cifar10_dataset(self):
        train_dataset = datasets.CIFAR10(self.data_path, train=True, download=True,
                                        transform=transforms.Compose(
                                            [
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ]
                                        )
        )
        val_dataset = datasets.CIFAR10(self.data_path, train=False, download=True, 
                                        transform=transforms.Compose(
                                            [
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ]
                                        )
        )
        
        return train_dataset, val_dataset

    def supervised_imagenet_dataset(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = ImagenetTrainDataset(self.data_path, transform=transform)
        val_dataset = ImagenetTestDataset(self.data_path, transform=transform)

        return train_dataset, val_dataset
    
    def self_supervised_stl10_dataset(self):
        transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(96), 2)
        train_dataset = datasets.STL10(self.data_path, split='train', download=True, transform=transform)
        val_dataset = datasets.STL10(self.data_path, split='test', download=True, transform=transform)

        return train_dataset, val_dataset

    def self_supervised_cifar10_dataset(self):
        transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(32), 2)
        train_dataset = datasets.CIFAR10(self.data_path, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(self.data_path, train=False, download=True, transform=transform)
        
        return train_dataset, val_dataset

    def self_supervised_imagenet_dataset(self):
        transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(224), 2)
        train_dataset = ImagenetTrainDataset(self.data_path, transform=transform)
        val_dataset = ImagenetTestDataset(self.data_path, transform=transform)

        return train_dataset, val_dataset

    def omniglot_dataset(self):
        # (self, mode='train', root='..' + os.sep + 'dataset', transform=None, target_transform=None, download=True)
        # TODO: more dataset setting, transform
        train_dataset = OmniglotDataset(mode='train', root=self.data_path)
        val_dataset = OmniglotDataset(mode='val', root=self.data_path)
        test_dataset = OmniglotDataset(mode='test', root=self.data_path)
        return train_dataset, val_dataset, test_dataset
    
    def miniimagenet_dataset(self):
        pass

    def cub_dataset(self):
        pass

    def awa1_dataset(self):
        pass

    def awa2_dataset(self):
        pass
    


if __name__=='__main__':
    names = os.listdir('./ImageNet/train')
    print(len(names))
    names = os.listdir('./ImageNet/val')
    print(len(names))
