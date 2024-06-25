import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from PIL import Image
from pathlib import Path
import albumentations as A
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from utils import BR_preprocessing
train_path = Path('/kaggle/input/rsna-bone-age/RSNA_Annotations/RSNA_Annotations/BONEAGE/boneage_train.csv')
val_path = Path('/kaggle/input/rsna-bone-age/RSNA_Annotations/RSNA_Annotations/BONEAGE/boneage_val.csv')
test_path = Path('/kaggle/input/rsna-bone-age/RSNA_Annotations/RSNA_Annotations/BONEAGE/gender_test.csv')

train_df = pd.read_csv(train_path, index_col='ID')
val_df = pd.read_csv(val_path, index_col='ID')
test_df = pd.read_csv(test_path, index_col='ID')

scaler = StandardScaler()
train_df['Boneage'] = scaler.fit_transform(train_df[['Boneage']])
val_df['Boneage'] = scaler.transform(val_df[['Boneage']])
train_df['Male'] = train_df['Male'].astype(float)
val_df['Male'] = val_df['Male'].astype(float)
test_df['Male'] = test_df['Male'].astype(float)

preprocessing = A.Compose([
    A.LongestMaxSize(max_size=500, interpolation=1),
    A.PadIfNeeded(min_height=500, min_width=500, border_mode=0, value=(0)),
    A.CLAHE(p=1.0),
    A.ToFloat(),
])

train_dir = '/kaggle/input/rsna-bone-age/RSNA_train/images'
train_ds = BR_preprocessing(train_df, train_dir, preprocessing)
val_dir = '/kaggle/input/rsna-bone-age/RSNA_val/images'
val_ds = BR_preprocessing(val_df, val_dir, preprocessing)
test_dir = '/kaggle/input/rsna-bone-age/RSNA_test/images'
test_ds = BR_preprocessing(test_df, test_dir, preprocessing, True)

datasets = [train_ds, val_ds, test_ds]
filenames = ['train_ds.pkl', 'val_ds.pkl', 'test_ds.pkl']
for dataset, filename in zip(datasets, filenames):
    subjects = []
    for subject in tqdm(dataset):
        subjects.append(subject)
    with open(filename, 'wb') as file:
        pkl.dump(subjects, file)
