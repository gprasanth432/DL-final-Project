
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

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(dl):
    model.train()
    loss_sum, correct, total = 0,0,0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*xb.size(0)
        preds = out.argmax(1)
        correct  += (preds==yb).sum().item()
        total    += xb.size(0)
    return loss_sum/total, correct/total

def eval_one_epoch(dl):
    model.eval()
    loss_sum, correct, total = 0,0,0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            loss_sum += loss.item()*xb.size(0)
            preds = out.argmax(1)
            correct  += (preds==yb).sum().item()
            total    += xb.size(0)
    return loss_sum/total, correct/total


num_epochs = 15
for e in range(1, num_epochs+1):
    tl, ta = train_one_epoch(train_loader)
    vl, va = eval_one_epoch(val_loader)
    print(f"Epoch {e} | Train Acc {ta:.4f}, Loss {tl:.4f} | Val Acc {va:.4f}, Loss {vl:.4f}")

torch.save(model.state_dict(), 'baseline_simplecnn.pt')
print("Saved baseline_simplecnn_weapons.pt")

import os

test_img_dir  = os.path.join(root_dir, 'test',  'images')
test_ann_dir  = os.path.join(root_dir, 'test',  'labels')

test_samples = parse_yolo(test_ann_dir, test_img_dir)

test_ds     = WeaponCropDataset(test_samples, tfms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print(f"Test samples: {len(test_ds)} → Test batches: {len(test_loader)}")

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_loss, test_acc = eval_one_epoch(test_loader)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(yb.tolist())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['gun','knife']))

print("Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['gun','knife'], yticklabels=['gun','knife'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

all_samples = train_samples + val_samples + test_samples
classes = [sample[-1] for sample in all_samples]

class_counts = Counter(classes)
class_names = {0: 'gun', 1: 'knife'}

print("Class Distribution:")
for cls, count in class_counts.items():
    print(f"{class_names[cls]}: {count}")

plt.figure(figsize=(6, 4))
sns.barplot(x=list(class_names.values()), y=list(class_counts.values()))
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.show()

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def display_samples(samples, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(samples))):
        p, x1, y1, x2, y2, lab = samples[i]
        img  = cv2.imread(p)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = img[y1:y2, x1:x2]

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(crop)
        plt.title(f"Class: {class_names[lab]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nSample 'gun' images:")
gun_samples = [s for s in all_samples if s[-1] == 0]
display_samples(gun_samples)

print("\nSample 'knife' images:")
knife_samples = [s for s in all_samples if s[-1] == 1]
display_samples(knife_samples)