import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class MonitorClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.image_files = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.jpg'):
                    self.image_files.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class MonitorDetectionDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.image_files = self.data['image_name'].tolist()
        self.quadrilateral = [torch.tensor([float(x) for x in pts.strip('[]').split(',')]) 
            for pts in self.data['points'].tolist()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.image_files[idx]))
        quadrilateral = torch.tensor(self.quadrilateral[idx])
        if self.transform:
            image = self.transform(image)
        return image, quadrilateral
