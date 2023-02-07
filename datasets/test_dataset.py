from torch.utils.data import Dataset
from PIL import Image
import os


class ScreenDataset(Dataset):
    def __init__(self, data_path, transform=None, is_dir=True):
        self.transform = transform
        self.is_dir = is_dir
        if not is_dir:
            self.image_path = data_path
            self.image = Image.open(self.image_path)
        else:
            self.folder_path = data_path
            self.image_paths = [os.path.join(self.folder_path, image_name) 
                for image_name in os.listdir(self.folder_path)]

    def __len__(self):
        return 1 if not self.is_dir else len(self.image_paths)

    def __getitem__(self, idx):
        if self.is_dir:
            image = Image.open(self.image_paths[idx])
        else:
            image = self.image
        if self.transform:
            image = self.transform(image)
        return image, None