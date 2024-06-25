import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from torchvision.models.convnext import LayerNorm2d
from torchvision.models import convnext_tiny
from torchmetrics.functional import mean_absolute_error as mae
import wandb
import os
from tqdm import tqdm
import time
import numpy as np
import transformers
from transformers import get_cosine_schedule_with_warmup
from model import BAR_model
from utils import BR_DS
from kaggle_secrets import UserSecretsClient

augmentation = A.Compose([
    A.HorizontalFlip(),
    A.GaussNoise(var_limit=0.005),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.05, 0.3)),
    A.Rotate(limit=10),
    A.Affine(shear=(-2, 2), scale=(0.95, 1.3), translate_percent=(0.01, 0.1), keep_ratio=True),
    ToTensorV2()
])

dir_ = '/kaggle/input/train_ds.pkl'
with open(dir_, 'rb') as file:
    train_list = pkl.load(file)
dir_ = '/kaggle/input/val_ds.pkl'
with open(dir_, 'rb') as file:
    val_list = pkl.load(file)

dir_ = '/kaggle/input/Scaler.pkl'
with open(dir_, 'rb') as file:
    standard_scaler = pkl.load(file)


train_ds = BR_DS(train_list[:2], augmentation)
val_ds = BR_DS(val_list[:2])

user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb")
wandb.login(key=wandb_key, verify=True)
torch.cuda.empty_cache()
wandb.init(
    project="BAR report",
    name='SWinT dummy run 3',
    config={
        "epochs": 20,
        "batch_size": 2,
        "init_lr": 1e-3, #initial_lr
        "dropout": 0.00,
        "drop_path":0.5
        })
config = wandb.config

# setup custom X-axis for epoch metrics
epoch_metrics = ['train_loss', 'val_loss']
wandb.define_metric('epoch')
for metric in epoch_metrics:
    wandb.define_metric(metric, step_metric='epoch') 

#config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = DataLoader(train_ds, batch_size = config.batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = config.batch_size, num_workers=4)
model = nn.DataParallel(BAR_model(config.dropout, config.drop_path)).to(device)
model = BAR_model(config.dropout, config.drop_path).to(device)
loss_fn = nn.L1Loss()
MODEL_DIR = '/kaggle/working/'
optimizer = torch.optim.AdamW(model.parameters(), config.init_lr, weight_decay=1e-8, amsgrad=True)
num_steps = round(len(train_ds) / config.batch_size + 1) * config.epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=num_steps)
scaler = torch.cuda.amp.GradScaler() # auto-mix precision for faster training
torch.backends.cudnn.benchmark = True #improved potential training speed at the cost of determinism, but stochasticity is ok here
torch.backends.cudnn.enabled = True

age_std = np.sqrt(standard_scaler.var_)[0]
age_mean = standard_scaler.mean_[0]
print (f'mean age:{age_mean}, age SD: {age_std}')
val_steps = len(val_ds) // config.batch_size + 1

#training loop
best_metric = 10
best_metric_epoch = -1
unfreeze_epoch = 5
t_step = 0
torch.cuda.empty_cache()
wandb.watch(model, log='all', log_freq=10)
for epoch in range(config.epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{config.epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for index, batch_data in tqdm(enumerate(train_loader)):
        inputs, sex, labels = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
        step += 1
        t_step += 1
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs, sex)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_lr = scheduler.get_last_lr()[0]
        step_loss = loss.item()
        step_metrics = {
            'step_loss':age_std * step_loss,
            'LR': step_lr,
        }
        epoch_loss += step_loss
        wandb.log(step_metrics)
    epoch_loss /= step
    
    #validation
    model.eval()
    metric = 0
    with torch.inference_mode():
        for val_data in val_loader:
            val_inputs, val_sex, val_labels = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
            val_outputs = model(val_inputs, val_sex)
            #val_metric
            metric += mae(val_outputs, val_labels).item()
    metric /= val_steps
    if metric < best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, f"best_metric_model.pth"),
        )
        print("saved new model")
    print(
        f"current MAE: {metric:.4f}"
        f"\nbest MAE: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
    )
    epoch_metrics = {
        "train_loss": age_std * epoch_loss, 
        "val_loss": age_std * metric,
        "epoch":epoch
    }
    wandb.log(epoch_metrics)
    print(f"duration of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
wandb.finish()