import os
import numpy as np
import torch
import argparse
import pandas as pd
import albumentations
import albumentations.pytorch

from glob import glob
from tqdm import tqdm
from prettyprinter import cpprint
from torch.utils.data import Dataset, DataLoader

from src.model import CNNModel
from src.utils import YamlConfigManager, seed_everything


def get_test_file_path(image_id):
    return f"E:/seti-breakthrough-listen/test/{image_id[0]}/{image_id}.npy"

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
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
        return image

def test(cfg):
    SEED = cfg.values.seed    
    BATCH_SIZE = cfg.values.batch_size    
    IMAGE_SIZE = cfg.values.image_size
    MODEL_ARC = cfg.values.model_arc
    NUM_CLASSES = cfg.values.num_classes
    MODEL_DIR = cfg.values.ckpt_dir
    USE_KFOLD = cfg.values.use_kfold
    NUM_FOLD = cfg.values.num_fold    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(SEED)

    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')

    test_df = pd.read_csv('E:/seti-breakthrough-listen/sample_submission.csv')    
    test_df['file_path'] = test_df['id'].apply(get_test_file_path)

    test_transform = albumentations.Compose([               
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        albumentations.pytorch.transforms.ToTensorV2()])

    test_dataset = TestDataset(df=test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNModel(model_arc=MODEL_ARC, num_classes=NUM_CLASSES)
    model.to(device)
    if USE_KFOLD:
        states = [torch.load(glob(MODEL_DIR + f'/{MODEL_ARC}/{k}_fold/*.pth')[0]) for k in range(1, NUM_FOLD + 1)]
    else:
        states = [torch.load(glob(MODEL_DIR + f'/{MODEL_ARC}/*.pth')[0])]

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    
    for i, (images) in progress_bar:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                logits = model(images)
            avg_preds.append(logits.sigmoid().to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)

    test_df['target'] = probs
    test_df[['id', 'target']].to_csv('submission.csv', index=False)
    test_df[['id', 'target']].head()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/eval_config.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()        

    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    test(cfg)