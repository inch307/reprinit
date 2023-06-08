from torchvision import datasets, transforms
import torch
import os
from PIL import Image
import scipy.io as sio
from imagenet_dataset import ImagenetTrainDataset, ImagenetTestDataset

class Dataset:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = self.args.model
        self.dataset = self.args.dataset
        self.data_path = kwargs['data_path']

    def get_dataset(self):
        if self.dataset == 'stl10':
            return self.supervised_stl10_dataset()
        elif self.dataset == 'imagenet':
            return self.supervised_imagenet_dataset()
        else:
            return self.supervised_cifar10_dataset()

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

        return train_dataset, val_dataset, None

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
        
        return train_dataset, val_dataset, None

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

        return train_dataset, val_dataset, None
 
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
