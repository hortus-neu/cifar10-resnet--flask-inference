'''
Author: Tao He
Date: 2025-09-07
'''

########################
### Basic Settings
########################

# --- Imports --- #
import os, time, random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

# --- Project paths (anchor to this file's folder) --- #
ROOT = Path(__file__).resolve().parent       # e.g. ~/cv-quickstart
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

# --- Global Constants --- #
# ImageNet mean/std normalization (commonly used for pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Dataset split
VAL_SIZE = 5000
SEED = 42

# Training parameters
BATCH_SIZE = 128
NUM_CLASSES = 10
# Phase 1 (warm-up, only train FC)
EPOCHS_WARMUP = 1
LR = 1e-3
WEIGHT_DECAY = 5e-4
# Phase 2 (fine-tune layer4 + FC)
EPOCHS_FINE = 15
FINE_LR = 1e-4

def unfreeze_layer4_and_fc(model):
    """
    Unfreeze only layer4 and fc, keep all other layers frozen.
    Set train/eval mode accordingly to stabilize BN statistics.
    """
    for p in model.parameters():
        p.requires_grad = False
    
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True
    
    model.eval()
    model.layer4.train()
    model.fc.train()
    return model

def trainable_parameters(model):
    """Return only trainable parameters (requires_grad=True) for the optimizer."""
    return (p for p in model.parameters() if p.requires_grad)

def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    """Select device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(num_classes=NUM_CLASSES, phase="head"):
    """
    Build ResNet18 model with pretrained ImageNet weights.
    
    phase="head": freeze backbone, only train the final FC layer.
    phase="all":  unfreeze and fine-tune more layers (or full model).
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    
    if phase == "head":
        for name, p in model.named_parameters():
            p.requires_grad = (name.startswith("fc"))
        model.eval()
        model.fc.train()
    else:
        for p in model.parameters():
            p.requires_grad = True
        model.train()
    
    return model

def get_transforms():
    """Define transforms for training and test datasets."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    return train_tf, test_tf

def accuracy(outputs, targets):
    """Compute accuracy = (#correct / total)."""
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    """Train for one epoch and log loss/accuracy."""
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        n += bs
    
    epoch_loss = running_loss / n
    epoch_acc = running_acc / n
    if writer is not None:
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Acc/train", epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, writer, epoch, split="val"):
    """Evaluate on validation set (or any split)."""
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        n += bs

    epoch_loss = running_loss / n
    epoch_acc = running_acc / n
    if writer is not None:
        writer.add_scalar(f"Loss/{split}", epoch_loss, epoch)
        writer.add_scalar(f"Acc/{split}", epoch_acc, epoch)
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_on_test(model, test_loader, device, best_path=os.path.join("weights", "resnet18_finetune_best.pt")):
    """Load best weights, run on test set, and plot confusion matrix."""
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f"✅ Test Accuracy: {acc:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=test_loader.dataset.classes)
    disp.plot(xticks_rotation=45, cmap="viridis")
    plt.title(f"Confusion Matrix (Test Accuracy={acc:.4f})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

def check_dataloader(loader, device, num_batches=1):
    """Debug helper: fetch a batch, print shape, and test device transfer."""
    for i, (images, labels) in enumerate(loader):
        print(f"[Batch {i}] images:", images.shape, "labels:", labels.shape)
        try:
            images = images.to(device)
            labels = labels.to(device)
            print("   moved to:", images.device, labels.device)
        except Exception as e:
            print("   ❌ move to device failed:", e)
        if i + 1 >= num_batches:
            break

def main():
    set_seed(SEED)
    device = get_device()
    if device.type == "mps":
        torch.set_float32_matmul_precision('high')
    print("device:", device)
    
    run_dir = LOGS_DIR / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    
    train_tf, test_tf = get_transforms()
    
    full_train = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_tf)

    train_size = len(full_train) - VAL_SIZE
    train_set, val_set = random_split(full_train, [train_size, VAL_SIZE], generator=torch.Generator().manual_seed(SEED))
    
    common_loader_args = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common_loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common_loader_args)
    
    writer = SummaryWriter(log_dir=str(run_dir))
    
    model = build_model(num_classes=NUM_CLASSES, phase="head").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Phase 1: Warm-up (train only FC)
    for epoch in range(EPOCHS_WARMUP):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        vl, va = evaluate(model, val_loader, criterion, device, writer, epoch, split="val")
        print(f"[Warmup {epoch}] train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")

    # Phase 2: Fine-tune (unfreeze layer4 + FC)
    os.makedirs("weights", exist_ok=True)
    best_path = WEIGHTS_DIR / "resnet18_finetune_best.pt"
    
    model = unfreeze_layer4_and_fc(model)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=FINE_LR, weight_decay=WEIGHT_DECAY)
    
    best_acc = 0.0
    for e in range(EPOCHS_WARMUP, EPOCHS_WARMUP + EPOCHS_FINE):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, e)
        vl, va = evaluate(model, val_loader, criterion, device, writer, e, split="val")
        print(f"[Fine {e}] train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")
        
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best! val_acc={best_acc:.4f} → saved to {best_path}")

    print(f"✅ Training finished! Best val_acc={best_acc:.4f}, weights saved to {best_path}")
    
    evaluate_on_test(model, test_loader, device, best_path)
    
    writer.close()

if __name__ == "__main__":
    main()
