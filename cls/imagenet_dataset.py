import torch
import os
import scipy.io as sio
from PIL import Image

# train
# n{label}_{id}.JPEG
# 1281167
# 1000 classes

# val
# validation ground truth
class ImagenetTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root + '/train/'
        self.transform = transform
        # self.label = self.get_label()

        names = os.listdir(self.data_root)
        mat_file = sio.loadmat(data_root + '/meta.mat')
        label_dict = {}
        for i in range(1000):
            label_dict[mat_file['synsets'][i][0][1][0]] = i

        self.data_list = []
        for name in names:
            id = name.split('_')[0]
            self.data_list.append((name, label_dict[id]))
        # print(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = Image.open(self.data_root + self.data_list[idx][0])
        x = x.convert('RGB')
        y = self.data_list[idx][1]
        if self.transform:
            x = self.transform(x)
        return x, y

class ImagenetTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root + '/val/'
        self.label_root = data_root
        self.names = sorted(os.listdir(self.data_root))
        self.transform = transform
        self.label = self.get_label()

    def __len__(self):
        return len(self.names)

    def get_label(self):
        labels = []
        ground_truth = open(self.label_root + '/ILSVRC2012_validation_ground_truth.txt', 'r')
        while True:
            line = ground_truth.readline()
            if not line: break
            labels.append(int(line)-1)
        ground_truth.close()
        
        return labels

    def __getitem__(self, idx):
        x = Image.open(self.data_root + self.names[idx])
        x = x.convert('RGB')
        y = self.label[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
