"""
src/cnn/train_cnn.py
====================
CNN Defect Classifier — ResNet18 / MobileNetV2 fine-tuning
Project : Manufacturing Defect Detection (Vision)
Team    : Baguio · Doton · Bernarte · De Castro
Course  : INTELSYS Final Project AY25-26

Overview
--------
This module fine-tunes a pre-trained CNN (ResNet18 or MobileNetV2) on the
Bridge Crack dataset to classify surface images as 'Crack' or 'Non-Crack'.

Pipeline:
  1. Load train/val/test splits from CSV (produced by the EDA notebook).
  2. Apply augmentation on train; resize+normalize on val/test.
  3. Replace the final classifier head to output 2 classes.
  4. Train with Adam + CosineAnnealingLR; evaluate every epoch.
  5. Log metrics (loss, accuracy, F1, false-alarm rate) to a CSV.
  6. Save best checkpoint by macro F1.

Usage:
  python src/cnn/train_cnn.py --model resnet18 --epochs 20 --batch 32
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CrackDataset(Dataset):
    """
    Loads images listed in a CSV file (columns: 'path', 'label').

    Parameters
    ----------
    csv_path  : str  – path to the split CSV (train.csv / val.csv / test.csv)
    transform : callable – torchvision transform pipeline
    """

    # Map string labels to integer class indices
    LABEL_MAP = {'Crack': 1, 'Non-Crack': 0}

    def __init__(self, csv_path: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = self.LABEL_MAP[row['label']]

        if self.transform:
            image = self.transform(image)

        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# 2. Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224):
    """
    Returns the appropriate torchvision transform pipeline.

    Training : resize → random horizontal/vertical flip → color jitter
               → random rotation → ToTensor → Normalize
    Val/Test  : resize → CenterCrop → ToTensor → Normalize
    """

    # These normalization values were computed from the dataset in the EDA notebook.
    # Replace with your actual computed values if they differ.
    mean = [0.485, 0.456, 0.406]  # ← update from EDA notebook output
    std  = [0.229, 0.224, 0.225]  # ← update from EDA notebook output

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Loads a pre-trained backbone and replaces the final classifier head.

    Supported architectures:
      - 'resnet18'      (ResNet-18, ~11M params)
      - 'mobilenet_v2'  (MobileNetV2, ~3.4M params — faster on CPU/edge)
    """

    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model   = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        # Replace the 1000-class ImageNet head with a 2-class head
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model   = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        # Replace final linear layer
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f'Unknown model: {model_name}. Choose resnet18 or mobilenet_v2.')

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4. Metrics helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds):
    """
    Computes macro F1, accuracy, and false-alarm rate.

    False-alarm rate = FP / (FP + TN) = rate at which Non-Crack images
    are incorrectly flagged as Crack. Minimizing this is a key project goal.
    """
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = (sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels))

    cm  = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # cm[0,0] = TN (Non-Crack predicted Non-Crack)
    # cm[0,1] = FP (Non-Crack predicted Crack)
    tn, fp = cm[0, 0], cm[0, 1]
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'f1': f1, 'acc': acc, 'false_alarm_rate': false_alarm_rate}


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full pass over the training set and returns avg loss."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)           # forward pass
        loss    = criterion(outputs, labels)
        loss.backward()                   # backprop
        optimizer.step()                  # weight update

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Runs evaluation on val/test set; returns loss + metrics dict."""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main training script
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = CrackDataset(args.train_csv, transform=get_transforms('train'))
    val_ds   = CrackDataset(args.val_csv,   transform=get_transforms('val'))
    test_ds  = CrackDataset(args.test_csv,  transform=get_transforms('test'))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=2)

    # ── Model, loss, optimizer, scheduler ─────────────────────────────────────
    model     = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()          # consider class weights if imbalanced
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # CosineAnnealing smoothly reduces LR → avoids abrupt LR drops
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Logging setup ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    log_path  = os.path.join(args.out_dir, 'train_log.csv')
    ckpt_path = os.path.join(args.out_dir, f'best_{args.model}.pth')

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss',
                         'val_f1', 'val_acc', 'val_far', 'lr'])

    best_f1 = 0.0

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss              = train_one_epoch(model, train_loader,
                                                  criterion, optimizer, device)
        val_loss, val_metrics   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f'Epoch [{epoch:3d}/{args.epochs}] '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Val F1: {val_metrics["f1"]:.4f} | '
            f'FAR: {val_metrics["false_alarm_rate"]:.4f} | '
            f'LR: {current_lr:.6f} | '
            f'Time: {elapsed:.1f}s'
        )

        # Log metrics
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss,
                             val_metrics['f1'], val_metrics['acc'],
                             val_metrics['false_alarm_rate'], current_lr])

        # Save best checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), ckpt_path)
            print(f'  ✓ New best F1 = {best_f1:.4f} — checkpoint saved.')

    # ── Final test evaluation ─────────────────────────────────────────────────
    print('\n── Test Set Evaluation ──')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f'  Test F1          : {test_metrics["f1"]:.4f}  (target ≥ 0.85)')
    print(f'  Test Accuracy    : {test_metrics["acc"]:.4f}')
    print(f'  False-Alarm Rate : {test_metrics["false_alarm_rate"]:.4f}  (target ≤ 0.05)')
    print(f'  Test Loss        : {test_loss:.4f}')
    print(f'\nLogs saved to  : {log_path}')
    print(f'Best model at  : {ckpt_path}')


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN defect classifier')
    parser.add_argument('--model',     type=str, default='resnet18',
                        choices=['resnet18', 'mobilenet_v2'],
                        help='Backbone architecture')
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--batch',     type=int,   default=32)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--train_csv', type=str,   default='data/splits/train.csv')
    parser.add_argument('--val_csv',   type=str,   default='data/splits/val.csv')
    parser.add_argument('--test_csv',  type=str,   default='data/splits/test.csv')
    parser.add_argument('--out_dir',   type=str,   default='outputs/cnn')
    args = parser.parse_args()
    main(args)
