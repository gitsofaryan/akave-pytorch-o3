#!/usr/bin/env python
"""
Full end-to-end training demo with real data upload, epoch logs, and CIDs.
Visualizes the complete PyTorch + Akave O3 workflow.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Load AKAVE_PRIVATE_KEY from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from pytorch_o3 import O3Client, O3CheckpointManager

# ============================================================================
# DEMO SETUP
# ============================================================================

print("\n" + "="*70)
print(" 🚀 PyTorch + Akave O3 Training Demo - Complete Workflow")
print("="*70 + "\n")

# Check for API key
api_key = os.getenv("AKAVE_PRIVATE_KEY")
if not api_key:
    print("⚠️  AKAVE_PRIVATE_KEY not set. Skipping O3 uploads (training will run locally).\n")
    USE_O3 = False
else:
    USE_O3 = True
    print(f"✅ AKAVE_PRIVATE_KEY configured (length: {len(api_key)})\n")

# ============================================================================
# SIMPLE CNN MODEL
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.adaptive(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ============================================================================
# LOAD SAMPLE DATA
# ============================================================================
print("📦 Loading sample dataset (MNIST)...\n")

sample_dir = Path("data/samples/mnist")
if not sample_dir.exists():
    print("❌ Sample dataset not found. Did you run data generation?")
    sys.exit(1)

train_data = torch.load(sample_dir / "train.pt", map_location="cpu", weights_only=True)
test_data = torch.load(sample_dir / "test.pt", map_location="cpu", weights_only=True)

train_images = train_data["images"].float() / 255.0
train_labels = train_data["labels"].long()
test_images = test_data["images"].float() / 255.0
test_labels = test_data["labels"].long()

# Add channel dimension if needed
if train_images.ndim == 3:
    train_images = train_images.unsqueeze(1)
if test_images.ndim == 3:
    test_images = test_images.unsqueeze(1)

print(f"   Training:   {train_images.shape[0]} samples, shape {list(train_images.shape[1:])}")
print(f"   Test:       {test_images.shape[0]} samples, shape {list(test_images.shape[1:])}\n")

# ============================================================================
# SETUP O3 (Optional)
# ============================================================================
o3_client = None
ckpt_manager = None
ckpt_bucket = "local-checkpoints"  # Default fallback

if USE_O3:
    try:
        print("🔌 Connecting to Akave O3...\n")
        o3_client = O3Client()
        
        # List available buckets
        buckets = o3_client.list_buckets()
        print(f"   Available buckets: {len(buckets)}")
        bucket_names = []
        for b in buckets[:5]:
            name = b.name if hasattr(b, 'name') else str(b)
            bucket_names.append(name)
            print(f"      - {name}")
        if len(buckets) > 5:
            print(f"      ... and {len(buckets) - 5} more")
            for b in buckets[5:]:
                name = b.name if hasattr(b, 'name') else str(b)
                bucket_names.append(name)
        print()
        
        # Setup checkpoint manager - use first available bucket
        if bucket_names:
            ckpt_bucket = bucket_names[0]
        else:
            ckpt_bucket = "pytorch-democheckpoints"
        try:
            # Use existing bucket for checkpoints
            ckpt_manager = O3CheckpointManager(o3_client, bucket_name=ckpt_bucket)
            print(f"✅ O3 Checkpoint Manager ready (bucket: {ckpt_bucket})\n")
        except Exception as e:
            print(f"⚠️  Could not setup checkpoint manager: {e}\n")
            ckpt_manager = None
    except Exception as e:
        print(f"⚠️  O3 connection failed: {e}")
        print("   Continuing with local training only...\n")
        o3_client = None

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("="*70)
print(" 🤖 Training Phase")
print("="*70 + "\n")

batch_size = 32
epochs = 3
learning_rate = 0.001
device = torch.device("cpu")

train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)

model = SimpleCNN(in_channels=train_images.shape[1], num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model: SimpleCNN with {total_params:,} parameters")
print(f"⚙️  Config: epochs={epochs}, batch={batch_size}, lr={learning_rate}\n")

# Create checkpoint directory
ckpt_dir = Path("data/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

checkpoints_saved = []

for epoch in range(1, epochs + 1):
    print("-" * 70)
    print(f"EPOCH {epoch}/{epochs}")
    print("-" * 70)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
        batch_count += 1

        if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0 or batch_idx == len(train_loader) - 1:
            avg_loss = running_loss / total
            train_acc = 100.0 * correct / total
            print(f"   Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%")

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(test_images.to(device))
        preds = outputs.argmax(dim=1)
        test_acc = 100.0 * preds.eq(test_labels.to(device)).sum().item() / len(test_labels)

    print(f"\n✅ Epoch {epoch} Summary:")
    print(f"   Training Loss:     {train_loss:.4f}")
    print(f"   Training Accuracy: {train_acc:.2f}%")
    print(f"   Test Accuracy:     {test_acc:.2f}%")

    # ── SAVE CHECKPOINT LOCALLY ──
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    ckpt_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
        "accuracy": test_acc,
    }
    torch.save(ckpt_payload, ckpt_path)
    ckpt_size = ckpt_path.stat().st_size
    print(f"\n   📁 Local checkpoint saved: {ckpt_path.name} ({ckpt_size / 1024:.1f} KB)")

    # ── UPLOAD TO O3 ──
    if ckpt_manager:
        try:
            print(f"\n   📤 Uploading to Akave O3...")
            import io
            buf = io.BytesIO()
            torch.save(ckpt_payload, buf)
            data_bytes = buf.getvalue()
            
            file_meta = o3_client.upload_object(
                ckpt_bucket,
                f"checkpoint_epoch_{epoch:03d}.pt",
                data_bytes
            )
            
            # Extract CID
            cid = None
            if hasattr(file_meta, 'root_cid'):
                cid = file_meta.root_cid
            elif hasattr(file_meta, 'RootCid'):
                cid = file_meta.RootCid
            elif isinstance(file_meta, dict):
                cid = file_meta.get('root_cid', file_meta.get('RootCid', file_meta.get('cid')))
            
            if cid:
                print(f"   ✅ O3 Upload Complete!")
                print(f"   🔗 CID (Content Identifier): {cid}")
                checkpoints_saved.append({
                    "epoch": epoch,
                    "cid": cid,
                    "accuracy": test_acc,
                    "loss": train_loss,
                })
            else:
                print(f"   ⚠️  Upload successful but CID extraction pending")
                checkpoints_saved.append({
                    "epoch": epoch,
                    "cid": "pending-cid",
                    "accuracy": test_acc,
                    "loss": train_loss,
                })
        except Exception as e:
            print(f"   ⚠️  O3 upload failed: {e}")
            checkpoints_saved.append({
                "epoch": epoch,
                "cid": "local-only",
                "accuracy": test_acc,
                "loss": train_loss,
            })
    else:
        checkpoints_saved.append({
            "epoch": epoch,
            "cid": "local-only",
            "accuracy": test_acc,
            "loss": train_loss,
        })
    
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print(" ✨ Training Complete!")
print("="*70 + "\n")

print("📋 Checkpoint Summary:\n")
print(f"{'Epoch':<8} {'Accuracy':<12} {'Loss':<10} {'CID / Location':<50}")
print("-" * 80)
for ckpt in checkpoints_saved:
    epoch = ckpt["epoch"]
    acc = ckpt["accuracy"]
    loss = ckpt["loss"]
    cid = ckpt["cid"]
    cid_display = cid[:47] + "..." if len(cid) > 50 else cid
    print(f"{epoch:<8} {acc:<12.2f}% {loss:<10.4f} {cid_display:<50}")

print()
best = max(checkpoints_saved, key=lambda c: c["accuracy"])
print(f"🏆 Best checkpoint: Epoch {best['epoch']} with {best['accuracy']:.2f}% accuracy")
print(f"   CID: {best['cid']}\n")

if ckpt_manager:
    print("✅ All checkpoints saved to:")
    print(f"   - Local: {ckpt_dir}/")
    print(f"   - O3 Bucket: {ckpt_bucket}/")
else:
    print("✅ Checkpoints saved locally to:", ckpt_dir)

print("\n" + "="*70 + "\n")
