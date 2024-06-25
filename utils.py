import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class BR_DS(Dataset):
    def __init__(self, list_, transform=None):
        self.list = list_
        self.transform = transform
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx):
        image, sex, y = self.list[idx][0], self.list[idx][1], self.list[idx][2]
        
        if self.transform:
            image = self.transform(image = image)
            image = image['image']
        else:
            image = torch.tensor(image).unsqueeze(0)
        return image, torch.tensor(sex, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    
class BR_preprocessing(Dataset): #Bone age regression dataset
    """
    Dataset class for Bone age preprocessing; performs resize, CLAHE and scaling to [0, 1]
    Args:
           X (DataFrame): DataFrame containing columns ID (index), Boneage and Male.
           
           root (string/Path): root directory for images
           
           transform: Albumentations Compose list.
           
           test: Binary indicator for test dataset; will not return a label value if True
    """
    def __init__(self, X, root, transform=None, test=False):
        self.X = X
        self.transform = transform
        self.root = root
        self.test = test
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = Image.open(self.root + '/' + str(self.X.iloc[idx].name)+'.png')
        if self.transform:
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            image = transformed['image']
            
        if not self.test:
            label = self.X.iloc[idx]['Boneage']
        male = self.X.iloc[idx]['Male']
        ID = self.X.index[idx]
        if self.test:
            return image, male, ID
        else:
            return image, male, label, ID
