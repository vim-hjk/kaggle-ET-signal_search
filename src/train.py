import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from adamp import AdamP
from madgrad import MADGRAD
from torch.cuda.amp import autocast, GradScaler

from .utils import get_score, AverageMeter, get_learning_rate, CosineAnnealingWarmupRestarts, SAM, remove_all_file
from .model import CNNModel


def train(cfg, train_loader, val_loader, val_labels, k):
    # Set Config
    MODEL_ARC = cfg.values.model_arc
    OUTPUT_DIR = cfg.values.output_dir
    NUM_CLASSES = cfg.values.num_classes

    SAVE_PATH = os.path.join(OUTPUT_DIR, MODEL_ARC)

    best_score = 0.

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(SAVE_PATH, exist_ok=True)

    if k > 0:
        os.makedirs(SAVE_PATH + f'/{k}_fold', exist_ok=True)

    num_epochs = cfg.values.train_args.num_epochs    
    max_lr = cfg.values.train_args.max_lr
    min_lr = cfg.values.train_args.min_lr
    weight_decay = cfg.values.train_args.weight_decay 
    log_intervals = cfg.values.train_args.log_intervals    

    model = CNNModel(model_arc=MODEL_ARC, num_classes=NUM_CLASSES)
    model.to(device)

    # base_optimizer = SGDP
    # optimizer = SAM(model.parameters(), base_optimizer, lr=max_lr, momentum=momentum)
    optimizer = MADGRAD(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    first_cycle_steps = len(train_loader) * num_epochs // 2

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=first_cycle_steps, 
        cycle_mult=1.0,
        max_lr=max_lr, 
        min_lr=min_lr, 
        warmup_steps=int(first_cycle_steps * 0.2), 
        gamma=0.5
    )

    criterion = nn.BCEWithLogitsLoss()

    wandb.watch(model)    

    for epoch in range(num_epochs):
        model.train()

        loss_values = AverageMeter()

        scaler = GradScaler()

        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            with autocast():
                logits = model(images)
                loss = criterion(logits.view(-1), labels)

            loss_values.update(loss.item(), batch_size)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            wandb.log({
                'Learning rate' : get_learning_rate(optimizer)[0],
                'Train Loss' : loss_values.val
            })

            if step % log_intervals == 0:
                tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}][{step}/{len(train_loader)}] || '
                           f'LR : {get_learning_rate(optimizer)[0]:.6e} || '
                           f'Train Loss : {loss_values.val:.4f} ({loss_values.avg:.4f}) ||')

        with torch.no_grad():
            model.eval()

            loss_values = AverageMeter()
            preds = []

            for step, (images, labels) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                logits = model(images)
                loss = criterion(logits.view(-1), labels)

                preds.append(logits.sigmoid().to('cpu').numpy())

                loss_values.update(loss.item(), batch_size)
        
        predictions = np.concatenate(preds)

        # f1, roc_auc = get_score(val_labels, predictions)
        roc_auc = get_score(val_labels, predictions)
        is_best = roc_auc >= best_score
        best_score = max(roc_auc, best_score)

        if is_best:
            if k > 0:
                remove_all_file(SAVE_PATH + f'/{k}_fold')
                print(f"Save checkpoints {SAVE_PATH + f'/{k}_fold/{epoch + 1}_epoch_{best_score * 100.0:.2f}%.pth'}...")
                torch.save(model.state_dict(), SAVE_PATH + f'/{k}_fold/{epoch + 1}_epoch_{best_score * 100.0:.2f}%.pth')
            else:
                remove_all_file(SAVE_PATH)
                print(f"Save checkpoints {SAVE_PATH + f'/{epoch + 1}_epoch_{best_score * 100.0:.2f}%.pth'}...")
                torch.save(model.state_dict(), SAVE_PATH + f'/{epoch + 1}_epoch_{best_score * 100.0:.2f}%.pth')

        wandb.log({            
            'Validation Loss average': loss_values.avg,
            'ROC AUC Score' : roc_auc,
            # 'F1 Score' : f1
        })

        tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}] || '
                   f'Val Loss : {loss_values.avg:.4f} || '
                   f'ROC AUC score : {roc_auc:.4f} ||')
                   # f'F1 score : {f1:.4f} ||')