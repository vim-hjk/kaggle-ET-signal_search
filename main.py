import os
from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
import numpy as np
import wandb
import torch
import argparse
import pandas as pd
import albumentations
import albumentations.pytorch

from prettyprinter import cpprint
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from src.train import train
from src.utils import YamlConfigManager, seed_everything, get_dataloader


def get_train_file_path(image_id):
        return f"E:/seti-breakthrough-listen/train/{image_id[0]}/{image_id}.npy"

def main(cfg):
    SEED = cfg.values.seed    
    BATCH_SIZE = cfg.values.train_args.batch_size    
    IMAGE_SIZE = cfg.values.image_size
    USE_KFOLD = cfg.values.use_kfold
    NUM_FOLD = cfg.values.train_args.num_fold if USE_KFOLD else 0
        

    seed_everything(SEED)

    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')

    data_df = pd.read_csv('E:/seti-breakthrough-listen/train_labels.csv') 

    data_df['file_path'] = data_df['id'].apply(get_train_file_path)

    train_transform = albumentations.Compose([               
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
        # albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        albumentations.pytorch.transforms.ToTensorV2()])

    val_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        albumentations.pytorch.transforms.ToTensorV2()])

    if USE_KFOLD:
        kfold = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED)

        for k, (train_index, val_index) in enumerate(kfold.split(data_df, data_df['target'])):
            print('\n')
            cpprint('=' * 15 + f'{k + 1}-Fold Cross Validation' + '=' * 15)
            train_df = data_df.iloc[train_index].reset_index(drop=True)
            val_df = data_df.iloc[val_index].reset_index(drop=True)

            train_loader = get_dataloader(df=train_df, transform=train_transform, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = get_dataloader(df=val_df, transform=val_transform, batch_size=BATCH_SIZE, shuffle=False)

            val_labels = val_df['target'].values.tolist()
            train(cfg, train_loader, val_loader, val_labels, k + 1)

    else:
        print('\n')
        cpprint('=' * 15 + f'Start Training' + '=' * 15)
        train_df, val_df = train_test_split(data_df, test_size=0.2, shuffle=True, stratify=data_df['target'], random_state=SEED)

        train_loader = get_dataloader(df=train_df, transform=train_transform, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = get_dataloader(df=val_df, transform=val_transform, batch_size=BATCH_SIZE, shuffle=False)

        val_labels = val_df['target'].values.tolist()
        train(cfg, train_loader, val_loader, val_labels, 0)


if __name__ == '__main__':
    wandb.init(project="kaggle-E.T.signal-classification", reinit=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/config.yml')
    parser.add_argument('--config', '-c', type=str, default='base')
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    wandb.run.name = cfg.values.model_arc
    wandb.run.save()
    wandb.config.update(cfg)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)