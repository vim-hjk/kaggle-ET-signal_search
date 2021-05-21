import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

class ETSignalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['target'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
        label = torch.tensor(self.labels[idx]).float()
        return image, label