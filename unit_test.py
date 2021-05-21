import cv2
import torch
import random
import numpy as np
import pandas as pd
import albumentations
import albumentations.pytorch

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


def get_train_file_path(image_id):
        return f"E:/seti-breakthrough-listen/train/{image_id[0]}/{image_id}.npy"

data_df = pd.read_csv('E:/seti-breakthrough-listen/train_labels.csv') 

data_df['file_path'] = data_df['id'].apply(get_train_file_path)

transform = albumentations.Compose([
    # albumentations.Resize(512, 512),
    albumentations.pytorch.ToTensorV2()
])

dataset = ETSignalDataset(df=data_df, transform=transform)

num = random.randint(0, 35000)

img, label = dataset.__getitem__(num)

img = img.permute(1, 2, 0).detach().cpu().numpy()

print(num, img.shape)
cv2.imshow(f'{label}_{num}', img)
cv2.waitKey(0)