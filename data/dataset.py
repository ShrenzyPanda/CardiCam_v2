import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import KFold


class MonitorClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True,
                num_splits=5, fold=0, seed=69):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.image_files = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if os.path.splitext(file)[1].lower() in ['.jpeg', '.jpg', '.png']:
                    self.image_files.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_idx[class_name])
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
        self.image_files, self.val_image_files, self.labels, self.val_labels = [],[],[],[]
        for i, (train_index, val_index) in enumerate(kf.split(self.image_files)):
            if i != fold:
                self.image_files.extend([self.image_files[x] for x in train_index])
                self.labels.extend([self.labels[x] for x in train_index])
            else:
                self.val_image_files.extend([self.image_files[x] for x in val_index])
                self.val_labels.extend([self.labels[x] for x in val_index])
        if not train:
            self.image_files = self.val_image_files
            self.labels = self.val_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class MonitorDetectionDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None, train=True, 
                num_splits=5, fold=0, seed=69):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.image_files = self.data['image_name'].tolist()
        self.quadrilateral = [torch.tensor([float(x) for x in pts.strip('[]').split(',')]) 
            for pts in self.data['points'].tolist()]
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
        self.image_files, self.val_image_files, self.quadrilateral, self.val_quadrilateral = [],[],[],[]
        for i, (train_index, val_index) in enumerate(kf.split(self.image_files)):
            if i != fold:
                self.image_files.extend([self.image_files[x] for x in train_index])
                self.quadrilateral.extend([self.quadrilateral[x] for x in train_index])
            else:
                self.val_image_files.extend([self.image_files[x] for x in val_index])
                self.val_quadrilateral.extend([self.quadrilateral[x] for x in val_index])
        if not train:
            self.image_files = self.val_image_files
            self.quadrilateral = self.val_quadrilateral
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.image_files[idx]))
        quadrilateral = torch.tensor(self.quadrilateral[idx])
        if self.transform:
            image = self.transform(image)
        return image, quadrilateral
