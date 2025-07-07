
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

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Class Distribution (Gun vs Knife)
labels = [lab for *_, lab in train_samples]
label_counts = Counter(labels)
label_names = {0: "Gun", 1: "Knife"}

plt.figure(figsize=(6, 4))
plt.bar(label_names.values(), [label_counts.get(k, 0) for k in label_names])
plt.title("Class Distribution in Weapons Training Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Crops")
plt.grid(True)
plt.show()

# Bounding Box Area Distribution
areas = []
for _, x1, y1, x2, y2, _ in train_samples:
    area = (x2 - x1) * (y2 - y1)
    if area > 0:
        areas.append(area)

plt.figure(figsize=(6, 4))
plt.hist(areas, bins=50, color='orange')
plt.title("Bounding Box Area Distribution")
plt.xlabel("Area (pixels²)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Aspect Ratio Distribution (Width / Height)
aspect_ratios = []
for _, x1, y1, x2, y2, _ in train_samples:
    w = x2 - x1
    h = y2 - y1
    if w > 0 and h > 0:
        aspect_ratios.append(w / h)

plt.figure(figsize=(6, 4))
plt.hist(aspect_ratios, bins=40, color='green')
plt.title("Bounding Box Aspect Ratios")
plt.xlabel("W / H")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Image Dimensions Distribution (W, H)
img_dims = []
for p, *_ in train_samples:
    img = cv2.imread(p)
    if img is not None:
        h, w = img.shape[:2]
        img_dims.append((w, h))

widths, heights = zip(*img_dims)
plt.figure(figsize=(6, 4))
plt.hist(widths, bins=30, alpha=0.6, label='Width')
plt.hist(heights, bins=30, alpha=0.6, label='Height')
plt.title("Original Image Dimensions")
plt.xlabel("Pixels")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Sample Grid of Weapon Crops (Gun and Knife)
import torchvision.utils
from torchvision import transforms
import torch

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

samples_by_class = {0: [], 1: []}
for sample in train_samples:
    p, x1, y1, x2, y2, lab = sample
    if lab in samples_by_class and len(samples_by_class[lab]) < 8:
        img = cv2.imread(p)
        if img is not None:
            crop = img[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                crop_tf = tfms(crop)
                samples_by_class[lab].append(crop_tf)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for i, cls in enumerate([0, 1]):
    if samples_by_class[cls]:
        grid = torchvision.utils.make_grid(samples_by_class[cls], nrow=4)
        axs[i].imshow(grid.permute(1, 2, 0))
        axs[i].set_title(label_names[cls])
        axs[i].axis('off')
plt.suptitle("Sample Weapon Crops")
plt.tight_layout()
plt.show()


import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

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

print("Batches -->", len(train_loader), "train |", len(val_loader), "val")
print("Total training samples:", len(train_ds))
print("Total validation samples:", len(val_ds))

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

history_baseline_tinycnn = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0.0
best_model_path = 'baseline_tinycnn.pt'

num_epochs = 15
for e in range(1, num_epochs + 1):
    tl, ta = train_one_epoch(train_loader)
    vl, va = eval_one_epoch(val_loader)

    # Save metrics to history
    history_baseline_tinycnn["train_loss"].append(tl)
    history_baseline_tinycnn["train_acc"].append(ta)
    history_baseline_tinycnn["val_loss"].append(vl)
    history_baseline_tinycnn["val_acc"].append(va)

    # Check for best model
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {e} with Val Acc: {va:.4f}")

    print(f"Epoch {e} | Train Acc {ta:.4f}, Loss {tl:.4f} | Val Acc {va:.4f}, Loss {vl:.4f}")

print(f"\nBest model saved to {best_model_path} with Val Acc: {best_val_acc:.4f}")

import os

test_img_dir  = os.path.join(root_dir, 'test',  'images')
test_ann_dir  = os.path.join(root_dir, 'test',  'labels')

test_samples = parse_yolo(test_ann_dir, test_img_dir)

test_ds     = WeaponCropDataset(test_samples, tfms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print(f"Test samples: {len(test_ds)} --> Test batches: {len(test_loader)}")

# Evaluate on Test Set & Compute Metrics
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
model = TinyCNN().to(device)
model.load_state_dict(torch.load("baseline_tinycnn.pt"))
model.eval()

test_loss, test_acc = eval_one_epoch(test_loader)
history_baseline_tinycnn["test_loss"] = (test_loss)
history_baseline_tinycnn["test_acc"] = (test_acc)
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
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

history_baseline_tinycnn["test_precision"] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
history_baseline_tinycnn["test_recall"] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
history_baseline_tinycnn["test_f1"] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
history_baseline_tinycnn["test_confusion_matrix"] = cm.tolist()

import matplotlib.pyplot as plt

train_loss = history_baseline_tinycnn["train_loss"]
val_loss   = history_baseline_tinycnn["val_loss"]
train_acc  = history_baseline_tinycnn["train_acc"]
val_acc    = history_baseline_tinycnn["val_acc"]
test_acc   = history_baseline_tinycnn.get("test_acc", None)

epochs = range(1, len(train_loss) + 1)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-o', label='Train Acc')
plt.plot(epochs, val_acc, 'r-o', label='Val Acc')
if test_acc is not None:
    plt.axhline(test_acc, color='g', linestyle='--', label=f'Test Acc = {test_acc:.2f}')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

!curl -L -o /content/human-parts-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/motsimaslam/human-parts-dataset

import zipfile
import os
import matplotlib.pyplot as plt
import cv2

human_parts_zip_path = '/content/human-parts-dataset.zip'
human_parts_extract_dir = '/content/data/human_parts_dataset'

# Unzip the human parts dataset
if not os.path.isdir(human_parts_extract_dir):
    with zipfile.ZipFile(human_parts_zip_path, 'r') as zf:
        zf.extractall(human_parts_extract_dir)
    print("Human parts dataset unzipped to:", human_parts_extract_dir)
else:
    print("Human parts dataset already unzipped at:", human_parts_extract_dir)

print("\nContents of the human parts dataset:")
for root, dirs, files in os.walk(human_parts_extract_dir):
    level = root.replace(human_parts_extract_dir, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]:
        print(f'{subindent}{f}')
    if len(files) > 5:
        print(f'{subindent}...')

print("\nVisualizing samples...")

image_paths = []
labels = []

for subdir, _, files in os.walk(human_parts_extract_dir):
    img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if img_files:
        sample_path = os.path.join(subdir, img_files[0])
        image_paths.append(sample_path)
        labels.append(os.path.basename(subdir))

n = len(image_paths)
cols = min(n, 5)
rows = (n + cols - 1) // cols

plt.figure(figsize=(15, 3 * rows))
for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(labels[i])
        plt.axis('off')

plt.tight_layout()
plt.suptitle("Sample Human Body Part Images", fontsize=16)
plt.show()

import json
import os

train_ann_sample_path = os.path.join(human_parts_extract_dir, 'train', 'ann', '456f2dc91bef78bd76b7af24e0f65314e2283a19.jpeg.json')
val_ann_sample_path   = os.path.join(human_parts_extract_dir, 'val',   'ann', 'e52d1397dfb1a29e77cf14b5af05f23354fa949b.jpeg.json')

def print_json_content(filepath, description):
    print(f"\n--- Content of {description} ({os.path.basename(filepath)}) ---")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2)[:1000] + "..." if len(json.dumps(data)) > 1000 else json.dumps(data, indent=2))
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filepath}")
        except Exception as e:
            print(f"An error occurred while reading {filepath}: {e}")
    else:
        print(f"File not found: {filepath}")


print_json_content(train_ann_sample_path, "Train annotation sample")
print_json_content(val_ann_sample_path, "Validation annotation sample")

human_parts_root_dir = human_parts_extract_dir
hp_train_img_dir     = os.path.join(human_parts_root_dir, 'train', 'img')
hp_train_ann_dir     = os.path.join(human_parts_root_dir, 'train', 'ann')
hp_val_img_dir       = os.path.join(human_parts_root_dir, 'val',   'img')
hp_val_ann_dir       = os.path.join(human_parts_root_dir, 'val',   'ann')

def parse_human_parts(ann_dir, img_dir):
    samples = []
    for fn in glob.glob(f"{ann_dir}/*.json"):
        base_fn = os.path.basename(fn)
        if base_fn.endswith('.json'):
            base_fn = base_fn[:-5]

        if base_fn.endswith('.jpeg'):
            base_fn = base_fn[:-5]

        img_fn = base_fn + '.jpeg'
        img_p  = os.path.join(img_dir, img_fn)

        if not os.path.isfile(img_p):
            print(f"Warning: Image file not found for annotation: {img_p}")
            continue

        try:
            with open(fn, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {fn}. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred reading {fn}: {e}. Skipping.")
            continue


        if 'objects' in data and isinstance(data['objects'], list):
            for obj in data['objects']:
                if (obj.get('geometryType') == 'rectangle' and
                    'points' in obj and 'exterior' in obj['points'] and
                    isinstance(obj['points']['exterior'], list) and
                    len(obj['points']['exterior']) == 2):

                    x1, y1 = obj['points']['exterior'][0]
                    x2, y2 = obj['points']['exterior'][1]
                    class_title = obj.get('classTitle', 'unknown')
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    samples.append((img_p, x1, y1, x2, y2, class_title))

    return samples

hp_train_samples = parse_human_parts(hp_train_ann_dir, hp_train_img_dir)
hp_val_samples   = parse_human_parts(hp_val_ann_dir, hp_val_img_dir)

print(f"Parsed Human Parts Crops: {len(hp_train_samples)} train | {len(hp_val_samples)} val")

human_parts_root_dir = human_parts_extract_dir

hp_train_img_dir     = os.path.join(human_parts_root_dir, 'train', 'img')
hp_train_ann_dir     = os.path.join(human_parts_root_dir, 'train', 'ann')
hp_val_img_dir       = os.path.join(human_parts_root_dir, 'val',   'img')
hp_val_ann_dir       = os.path.join(human_parts_root_dir, 'val',   'ann')

hp_train_samples = parse_human_parts(hp_train_ann_dir, hp_train_img_dir)
hp_val_samples   = parse_human_parts(hp_val_ann_dir, hp_val_img_dir)

# Print the number of samples found
print(f"Parsed Human Parts Crops: {len(hp_train_samples)} train | {len(hp_val_samples)} val")

import os
import json
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from collections import Counter

# seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class EnhancedWeaponCropDataset(Dataset):

    def __init__(self, weapon_samples, human_parts_samples=None, transform=None,
                 balance_classes=True, max_human_parts=None):
        self.samples = []
        self.transform = transform

        for sample in weapon_samples:
            self.samples.append(sample)

        # Add human parts samples as class 2 (negative samples)
        if human_parts_samples:
            human_parts_list = list(human_parts_samples)

            if max_human_parts and len(human_parts_list) > max_human_parts:
                human_parts_list = random.sample(human_parts_list, max_human_parts)

            for img_path, x1, y1, x2, y2, class_title in human_parts_list:
                self.samples.append((img_path, x1, y1, x2, y2, 2))

        if balance_classes:
            self._balance_classes()

        print(f"Dataset created with {len(self.samples)} samples")
        self._print_class_distribution()

    def _balance_classes(self):
        class_samples = {0: [], 1: [], 2: []}
        for sample in self.samples:
            class_id = sample[5]
            if class_id in class_samples:
                class_samples[class_id].append(sample)

        min_size = min(len(samples) for samples in class_samples.values() if len(samples) > 0)

        balanced_samples = []
        for class_id, samples in class_samples.items():
            if len(samples) > 0:
                if len(samples) > min_size:
                    samples = random.sample(samples, min_size)
                balanced_samples.extend(samples)

        self.samples = balanced_samples
        print(f"Balanced dataset to {len(self.samples)} samples")

    def _print_class_distribution(self):
        class_counts = Counter(sample[5] for sample in self.samples)
        print("Class distribution:")
        class_names = {0: 'gun', 1: 'knife', 2: 'human_part'}
        for class_id, count in sorted(class_counts.items()):
            print(f"  {class_names.get(class_id, f'class_{class_id}')}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x1, y1, x2, y2, class_id = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return torch.zeros(3, 32, 32), 0

        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
        else:
            return torch.zeros(3, 32, 32), 0

        # Apply transformations
        if self.transform:
            crop = self.transform(crop)

        return crop, class_id

def create_enhanced_datasets_for_existing_code(train_samples, val_samples, test_samples,
                                               hp_train_samples, hp_val_samples):
    train_transforms, val_transforms = create_enhanced_transforms()

    # Create enhanced datasets
    train_ds = EnhancedWeaponCropDataset(
        weapon_samples=train_samples,
        human_parts_samples=hp_train_samples,
        transform=train_transforms,
        balance_classes=True,
        max_human_parts=8000  # Adjust based on your memory constraints
    )

    val_ds = EnhancedWeaponCropDataset(
        weapon_samples=val_samples,
        human_parts_samples=hp_val_samples,
        transform=val_transforms,
        balance_classes=True,
        max_human_parts=2000
    )

    test_ds = EnhancedWeaponCropDataset(
        weapon_samples=test_samples,
        human_parts_samples=None,  # No human parts in test set
        transform=val_transforms,
        balance_classes=False
    )

    return train_ds, val_ds, test_ds

class ImprovedTinyCNN(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_enhanced_transforms():
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms

def train_one_epoch_enhanced(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

def validate_enhanced(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy, all_preds, all_targets

num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ImprovedTinyCNN(num_classes=3).to(device)
print("Model architecture:")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=3, embed_dim=96, depth=4, num_heads=3):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2, embed_dim))
        self.pos_drop = nn.Dropout(0.)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))

import os
import cv2
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EnhancedWeaponCropDataset(Dataset):
    def __init__(self, weapon_samples, human_parts_samples=None, transform=None,
                 balance_classes=True, max_human_parts=None):
        self.samples = []
        self.transform = transform

        self.samples.extend(weapon_samples)

        if human_parts_samples:
            if max_human_parts and len(human_parts_samples) > max_human_parts:
                human_parts_samples = random.sample(human_parts_samples, max_human_parts)
            for img_path, x1, y1, x2, y2, _ in human_parts_samples:
                self.samples.append((img_path, x1, y1, x2, y2, 2))  # class_id=2 for human_part

        if balance_classes:
            self._balance_classes()
        self._print_class_distribution()

    def _balance_classes(self):
        class_samples = {0: [], 1: [], 2: []}
        for sample in self.samples:
            class_id = sample[5]
            class_samples[class_id].append(sample)

        min_len = min(len(v) for v in class_samples.values() if v)
        balanced = []
        for v in class_samples.values():
            balanced.extend(random.sample(v, min_len))
        self.samples = balanced

    def _print_class_distribution(self):
        dist = Counter([s[5] for s in self.samples])
        for cid in sorted(dist.keys()):
            print(f"Class {cid}: {dist[cid]} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x1, y1, x2, y2, class_id = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, 32, 32), 0
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((32, 32, 3), dtype=np.uint8)
        if self.transform:
            crop = self.transform(crop)
        return crop, class_id

def create_enhanced_transforms():
    train_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tfms, val_tfms

def train_one_epoch_mixup(model, dataloader, criterion, optimizer, device, mixup_alpha=0.4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (
            lam * predicted.eq(targets_a.data).sum().item() +
            (1 - lam) * predicted.eq(targets_b.data).sum().item()
        )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

import pickle

history_mixup_tinyvit = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

best_val_acc = 0.0
best_model_path = "mixup_tinyvit_with_negatives.pt"
patience = 5
epochs_no_improve = 0

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch_mixup(
        model, train_loader, criterion, optimizer, device, mixup_alpha=0.4
    )

    val_loss, val_acc, val_preds, val_targets = validate_enhanced(
        model, val_loader, criterion, device
    )

    history_mixup_tinyvit["train_loss"].append(train_loss)
    history_mixup_tinyvit["train_acc"].append(train_acc)
    history_mixup_tinyvit["val_loss"].append(val_loss)
    history_mixup_tinyvit["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch} | Saved best model with val acc {val_acc:.4f}")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch} | No improvement. {epochs_no_improve}/{patience} early stop patience.")

    scheduler.step()
    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
        break

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

class_names = ['gun', 'knife', 'human_part']
print("\nEvaluating model on test set...")

test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

test_loss, test_acc, test_preds, test_targets = validate_enhanced(
    model, test_loader, criterion, device
)

precision, recall, f1, support = precision_recall_fscore_support(
    test_targets, test_preds, average=None, labels=[0, 1, 2], zero_division=0
)
cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])

print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(
    test_targets, test_preds, target_names=class_names, zero_division=0
))
print("\nConfusion Matrix:")
print(cm)

history_mixup_tinyvit["test_loss"] = test_loss
history_mixup_tinyvit["test_acc"] = test_acc
history_mixup_tinyvit["test_precision"] = precision
history_mixup_tinyvit["test_recall"] = recall
history_mixup_tinyvit["test_f1"] = f1
history_mixup_tinyvit["test_confusion_matrix"] = cm

import matplotlib.pyplot as plt

train_loss = history_mixup_tinyvit["train_loss"]
val_loss   = history_mixup_tinyvit["val_loss"]
train_acc  = history_mixup_tinyvit["train_acc"]
val_acc    = history_mixup_tinyvit["val_acc"]
test_acc   = history_mixup_tinyvit.get("test_acc", None)

epochs = range(1, len(train_loss) + 1)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-o', label='Train Acc')
plt.plot(epochs, val_acc, 'r-o', label='Val Acc')
if test_acc is not None:
    plt.axhline(test_acc, color='g', linestyle='--', label=f'Test Acc = {test_acc:.2f}')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""## training vit with mixup and weight decay changes"""

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import pickle

def mixup_data(x, y, alpha=0.4, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch_mixup(model, loader, criterion, optimizer, device, mixup_alpha=0.4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(1)
        correct += (lam * preds.eq(y_a).sum().item() + (1 - lam) * preds.eq(y_b).sum().item())
        total += y.size(0)
    return total_loss / total, correct / total

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def validate_enhanced(model, loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            preds = outputs.argmax(1)
            loss_total += loss.item() * x.size(0)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return loss_total / total, correct / total, all_preds, all_targets

def comprehensive_evaluation(model, loader, device, class_names):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    return accuracy, precision, recall, f1, cm

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 30
mixup_alpha = 0.4
best_val_acc = 0.0
patience = 5
epochs_no_improve = 0
best_model_path = "mixup_wd_tinyvit_with_negatives.pt"

history_mixup_wd_tinyvit = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, mixup_alpha)
    val_loss, val_acc, _, _ = validate_enhanced(model, val_loader, criterion, device)

    history_mixup_wd_tinyvit["train_loss"].append(train_loss)
    history_mixup_wd_tinyvit["train_acc"].append(train_acc)
    history_mixup_wd_tinyvit["val_loss"].append(val_loss)
    history_mixup_wd_tinyvit["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch} | Saved best model with val acc {val_acc:.4f}")

    scheduler.step()
    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
        break

model.load_state_dict(torch.load("/content/mixup_wd_tinyvit_with_negatives.pt"))
model.eval()

test_loss, test_acc, test_preds, test_targets = validate_enhanced(model, test_loader, criterion, device)

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

class_names = ['gun', 'knife', 'human_part']

precision, recall, f1, support = precision_recall_fscore_support(
    test_targets, test_preds, average=None, labels=[0, 1, 2], zero_division=0
)
report = classification_report(test_targets, test_preds, target_names=class_names, digits=4, zero_division=0)
cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])

print(f"\n=== TEST EVALUATION ===")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)


history_mixup_wd_tinyvit["test_loss"] = test_loss
history_mixup_wd_tinyvit["test_acc"] = test_acc
history_mixup_wd_tinyvit["test_precision"] = precision
history_mixup_wd_tinyvit["test_recall"] = recall
history_mixup_wd_tinyvit["test_f1"] = f1
history_mixup_wd_tinyvit["test_confusion_matrix"] = cm

print(history_mixup_wd_tinyvit)

import matplotlib.pyplot as plt

train_loss = history_mixup_wd_tinyvit["train_loss"]
val_loss   = history_mixup_wd_tinyvit["val_loss"]
train_acc  = history_mixup_wd_tinyvit["train_acc"]
val_acc    = history_mixup_wd_tinyvit["val_acc"]
test_acc   = history_mixup_wd_tinyvit.get("test_acc", None)

epochs = range(1, len(train_loss) + 1)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-o', label='Train Acc')
plt.plot(epochs, val_acc, 'r-o', label='Val Acc')
if test_acc is not None:
    plt.axhline(test_acc, color='g', linestyle='--', label=f'Test Acc = {test_acc:.2f}')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CustomResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_classes = 3  # 'gun', 'knife', 'human_part'
model_ft = CustomResNet18(num_classes=num_classes).to(device)
print(f"Loaded custom-defined ResNet18 model with {num_classes} output classes.")

for name, param in model_ft.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

print("\nFroze feature extractor layers. Only classifier head will be trained initially.")

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)

print("Defined CrossEntropyLoss and Adam optimizer for the classifier head.")

!pip install torchinfo

from torchinfo import summary

summary(model_ft, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])

import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def train_one_epoch_enhanced(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy

def validate_enhanced(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy, all_preds, all_targets

num_epochs_head = 5
print(f"\nStarting initial training (classifier head only) for {num_epochs_head} epochs...")

history_resnet18 = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

best_val_acc = 0.0
patience = 5
epochs_no_improve = 0
best_model_path = "resnet18_best_head_finetune.pt"

for epoch in range(1, num_epochs_head + 1):
    train_loss, train_acc = train_one_epoch_enhanced(
        model_ft, train_loader, criterion, optimizer_ft, device
    )

    val_loss, val_acc, _, _ = validate_enhanced(
        model_ft, val_loader, criterion, device
    )

    history_resnet18["train_loss"].append(train_loss)
    history_resnet18["train_acc"].append(train_acc)
    history_resnet18["val_loss"].append(val_loss)
    history_resnet18["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model_ft.state_dict(), best_model_path)
        print(f"Epoch {epoch} | Saved best model with val acc {val_acc:.4f}")

    print(f"Epoch {epoch}/{num_epochs_head} (Head Only) | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

print("\nFinished initial training of classifier head.")

print("\nUnfreezing all model layers for fine-tuning...")
for param in model_ft.parameters():
    param.requires_grad = True

optimizer_fine_tune = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler_fine_tune = optim.lr_scheduler.StepLR(optimizer_fine_tune, step_size=10, gamma=0.1)

num_epochs_fine_tune = 15
print(f"\nStarting fine-tuning (entire model) for {num_epochs_fine_tune} epochs...")

for epoch in range(1, num_epochs_fine_tune + 1):
    train_loss, train_acc = train_one_epoch_enhanced(
        model_ft, train_loader, criterion, optimizer_fine_tune, device
    )

    val_loss, val_acc, _, _ = validate_enhanced(
        model_ft, val_loader, criterion, device
    )

    history_resnet18["train_loss"].append(train_loss)
    history_resnet18["train_acc"].append(train_acc)
    history_resnet18["val_loss"].append(val_loss)
    history_resnet18["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model_ft.state_dict(), best_model_path)
        print(f"Epoch {epoch} | Saved best model with val acc {val_acc:.4f}")

    scheduler_fine_tune.step()

    print(f"Epoch {epoch}/{num_epochs_fine_tune} (Fine-tuning) | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
        break

import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

best_model_path = "resnet18_best_head_finetune.pt"


model_ft.load_state_dict(torch.load(best_model_path))
model_ft.eval()
print(f"\nLoaded best model from {best_model_path} for final test evaluation.")

print("\nRunning final evaluation on test set...")
test_loss, test_acc, test_preds, test_targets = validate_enhanced(
    model_ft, test_loader, criterion, device
)

precision = precision_score(test_targets, test_preds, average="weighted", zero_division=0)
recall = recall_score(test_targets, test_preds, average="weighted", zero_division=0)
f1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)
cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])

print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
print("Test Confusion Matrix:\n", cm)

history_resnet18["test_loss"] = test_loss
history_resnet18["test_acc"] = test_acc
history_resnet18["test_precision"] = precision
history_resnet18["test_recall"] = recall
history_resnet18["test_f1"] = f1
history_resnet18["test_confusion_matrix"] = cm.tolist()  # Store as list for JSON compatibility

import matplotlib.pyplot as plt

train_loss = history_resnet18["train_loss"]
val_loss   = history_resnet18["val_loss"]
train_acc  = history_resnet18["train_acc"]
val_acc    = history_resnet18["val_acc"]
test_acc   = history_resnet18.get("test_acc", None)

epochs = range(1, len(train_loss) + 1)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-o', label='Train Acc')
plt.plot(epochs, val_acc, 'r-o', label='Val Acc')
if test_acc is not None:
    plt.axhline(test_acc, color='g', linestyle='--', label=f'Test Acc = {test_acc:.2f}')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

models_histories = {
    "baseline_tinycnn": history_baseline_tinycnn,
    "enhanced_tinycnn": history_enhanced_tinycnn,
    "tinyvit": history_tinyvit,
    "tuned_tinyvit": history_tuned_tinyvit,
    "mixup_tinyvit": history_mixup_tinyvit,
    "mixup_wd_tinyvit": history_mixup_wd_tinyvit,
    "resnet18": history_resnet18
}

metrics_dict = {}
for model_name, hist in models_histories.items():
    metrics_dict[model_name] = {
        "Test Accuracy": hist.get("test_acc", None),
        "Test Loss": hist.get("test_loss", None),
        "F1-Score": hist.get("test_f1", None),
        "Precision": hist.get("test_precision", None),
        "Recall": hist.get("test_recall", None)
    }

import pandas as pd
import matplotlib.pyplot as plt

df_metrics = pd.DataFrame(metrics_dict).T
display(df_metrics.round(4))

df_metrics[["Test Accuracy", "F1-Score", "Precision", "Recall"]].plot(kind='bar', figsize=(12,6))
plt.title("Model Comparison based on Saved History")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

models_histories = {
    "baseline_tinycnn": history_baseline_tinycnn,
    "enhanced_tinycnn": history_enhanced_tinycnn,
    "tinyvit": history_tinyvit,
    "tuned_tinyvit": history_tuned_tinyvit,
    "mixup_tinyvit": history_mixup_tinyvit,
    "mixup_wd_tinyvit": history_mixup_wd_tinyvit,
    "resnet18": history_resnet18
}

metrics_dict = {}
for model_name, hist in models_histories.items():
    metrics_dict[model_name] = {
        "Test Accuracy": hist.get("test_acc", None),
        "Test Loss": hist.get("test_loss", None),
        "F1-Score": np.mean(hist.get("test_f1", [0])),
        "Precision": np.mean(hist.get("test_precision", [0])),
        "Recall": np.mean(hist.get("test_recall", [0]))
    }

import pandas as pd
import matplotlib.pyplot as plt

df_metrics = pd.DataFrame(metrics_dict).T
display(df_metrics.round(4))  # Show the table

plot_titles = {
    "Test Accuracy": "Model Comparison - Test Accuracy",
    "Test Loss": "Model Comparison - Test Loss",
    "F1-Score": "Model Comparison - F1 Score",
    "Precision": "Model Comparison - Precision",
    "Recall": "Model Comparison - Recall"
}

for metric in plot_titles:
    plt.figure(figsize=(10, 5))
    df_metrics[metric].plot(kind='bar', color='skyblue')
    plt.title(plot_titles[metric])
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    if metric != "Test Loss":
        plt.ylim(0, 1.05)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

metrics_to_plot = ["Test Accuracy", "F1-Score", "Precision", "Recall"]
df_metrics[metrics_to_plot].plot(kind='bar', figsize=(12, 6))
plt.title("Model Comparison on Key Evaluation Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()