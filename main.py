from google.colab import drive
import zipfile, os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob, cv2

drive.mount('/content/drive')

zip_path    = '/content/drive/MyDrive/knivesnpistols.zip'
extract_dir = '/content/data/knivesnpistols'

if not os.path.isdir(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
print("Unzipped to:", extract_dir)

root_dir      = os.path.join(extract_dir, 'combined_gunsnknifes')
train_img_dir = os.path.join(root_dir, 'train', 'images')
train_ann_dir = os.path.join(root_dir, 'train', 'labels')
val_img_dir   = os.path.join(root_dir, 'val',   'images')
val_ann_dir   = os.path.join(root_dir, 'val',   'labels')

print(f"Train: {len(os.listdir(train_img_dir))} imgs, {len(os.listdir(train_ann_dir))} labels")
print(f" Val:  {len(os.listdir(val_img_dir))} imgs, {len(os.listdir(val_ann_dir))} labels")


def parse_yolo(ann_dir, img_dir):
    samples = []
    for fn in glob.glob(f"{ann_dir}/*.txt"):
        img_fn = os.path.basename(fn).replace('.txt','.jpg')
        img_p  = os.path.join(img_dir, img_fn)
        if not os.path.isfile(img_p): continue
        h,w = cv2.imread(img_p).shape[:2]
        for line in open(fn):
            cls, xc, yc, wr, hr = map(float, line.split())
            x1 = int((xc - wr/2)*w); y1 = int((yc - hr/2)*h)
            x2 = int((xc + wr/2)*w); y2 = int((yc + hr/2)*h)
            samples.append((img_p, x1, y1, x2, y2, int(cls)))
    return samples

train_samples = parse_yolo(train_ann_dir, train_img_dir)
val_samples   = parse_yolo(val_ann_dir,   val_img_dir)
print("Crops:", len(train_samples), "train |", len(val_samples), "val")

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

class WeaponCropDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, x1, y1, x2, y2, lab = self.samples[idx]
        img  = cv2.imread(p)
        crop = img[y1:y2, x1:x2]
        if self.transform:
            crop = self.transform(crop)
        return crop, lab

train_ds = WeaponCropDataset(train_samples, tfms)
val_ds   = WeaponCropDataset(val_samples,   tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

print("Batches →", len(train_loader), "train |", len(val_loader), "val")

import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*16*16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = TinyCNN().to(device)
print(model)