import pickle
import struct
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class Cifar100(Dataset):
    def __init__(self, img_gzip, label_name_zip, base_dir, transform=False):
        self.gzip_path = base_dir + img_gzip
        self.label_name_gzip = base_dir + label_name_zip
        self.imgs, self.labels, self.label_name = [], [], []
        self.transform = transform
        self.transformation =  v2.Compose([v2.RandomResizedCrop(size=(32, 32), antialias=True),
                                            v2.RandomHorizontalFlip(p=0.5)])
        self.classes = None
        self.load_label()
        self.load()
    
    def load_label(self):
        with open(self.label_name_gzip,'rb') as f:
            data = pickle.load(f)
            self.label_name = data['fine_label_names']
            self.classes = len(self.label_name)
        
    def load(self):
        with open(self.gzip_path,'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels= data['fine_labels']
        self.imgs = self.imgs.reshape(-1, 3, 32, 32)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32)/255.0
        if self.transform : img = self.transformation(img)
        label = torch.tensor(self.labels[idx])
        return img, label