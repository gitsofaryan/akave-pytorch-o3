## PyTorch + Akave O3 Integration

**Decentralized ML training pipeline** — stream datasets, train models, and store CID-based immutable checkpoints on Akave O3 storage.

### 🎯 At a Glance
| Component | Purpose |
|-----------|---------|
| **O3Client** | Thin wrapper around `akavesdk` for streaming, range downloads, uploads with CID return |
| **O3Dataset** | PyTorch `Dataset` that streams samples from O3 with two-tier caching (LRU + disk) |
| **O3CheckpointManager** | CID-versioned checkpoint persistence with lineage tracking and auto-resume |
| **Streamlit Dashboard** | GUI for dataset management, training, checkpoint versioning with real-time logs |
| **MNIST Example** | End-to-end training: `examples/train_mnist.py` |

### ✨ Key Features
- ✅ **Content-addressed versioning**: Every checkpoint gets a unique CID (immutable hash)
- ✅ **Auto-resume**: Detect latest checkpoint, continue from that epoch
- ✅ **Real-time logs**: Monitor loss, accuracy, batch progress during training
- ✅ **Rate-limit resilience**: Exponential backoff + retry logic for large uploads
- ✅ **Multiprocessing-safe**: Per-worker O3Client instances in DataLoader
- ✅ **Local + O3 storage**: Train locally, upload to O3 automatically

---

## 🚀 Quick Start

### 1. Setup (5 minutes)
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows: PowerShell
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set your private key
echo AKAVE_PRIVATE_KEY="your_64_hex_chars" > .env
```

### 2. Option A: CLI Training (fastest)
```bash
python examples/train_mnist.py \
  --o3-data-bucket mnist-data \
  --o3-checkpoint-bucket mnist-checkpoints \
  --epochs 5
```

### 2. Option B: Streamlit Dashboard (recommended)
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

Then:
1. Go to **Settings** → enter AKAVE_PRIVATE_KEY (or paste from .env)
2. Go to **Dashboard** → select a dataset
3. Click **▶ Start Training**
4. Watch real-time logs, loss, accuracy
5. View checkpoints + CIDs on **Checkpoints** page

### 📦 Core Dependencies
- `torch>=2.0.0` — PyTorch
- `akavesdk` — Akave O3 SDK (from Git)
- `tenacity>=8.2.0` — Retry logic
- `streamlit>=1.0.0` — Dashboard (optional)
- `plotly` — Charts (optional)

---

## 🔑 Configuration

### Setting AKAVE_PRIVATE_KEY
The library requires your **64-character hex private key** for O3 authentication.

**Option 1: Environment file** (recommended)
```bash
# Create .env file
echo AKAVE_PRIVATE_KEY="your_64_hex_chars" > .env

# The library auto-loads from .env
python examples/train_mnist.py ...
```

**Option 2: Export environment variable**
```bash
# macOS/Linux
export AKAVE_PRIVATE_KEY="your_64_hex_chars"

# Windows PowerShell
$env:AKAVE_PRIVATE_KEY = "your_64_hex_chars"
```

**Option 3: Direct initialization**
```python
from pytorch_o3 import O3Client
client = O3Client(private_key="your_64_hex_chars")
```

⚠️ **Important**: Missing key → `O3AuthError`. Check setup before training.

---

## 📚 Usage Guides

### CLI: Train MNIST with O3
```bash
python examples/train_mnist.py \
  --o3-data-bucket mnist-data \
  --o3-train-prefix mnist/train/ \
  --o3-test-prefix mnist/test/ \
  --o3-checkpoint-bucket mnist-checkpoints \
  --epochs 5 \
  --batch-size 32 \
  --lr 0.001
```

**Key Arguments**:
- `--o3-data-bucket` (required) — Bucket with training/test objects
- `--o3-train-prefix` (default: `mnist/train/`) — Training data location
- `--o3-test-prefix` (default: `mnist/test/`) — Test data location
- `--o3-checkpoint-bucket` (required) — Where to store checkpoints
- `--epochs`, `--batch-size`, `--lr` — Standard training controls

**What happens each epoch**:
1. Stream batches from O3 via `O3Dataset`
2. Train model, evaluate on test set
3. Save checkpoint (.pt + metadata JSON) to O3
4. Log CID for later reference
5. Auto-resume persists via CID-based lineage

### GUI: Streamlit Dashboard
```bash
streamlit run app.py
```

**Pages**:
| Page | Purpose |
|------|----------|
| **Overview** | Quick start, architecture diagram, key concepts |
| **Dashboard** | Dataset selection, training config, real-time progress |
| **Datasets** | Browse bundled datasets, preview tensors |
| **Training** | Train job status, live logs, checkpoint summary |
| **Checkpoints** | All checkpoints, CID lineage graph, resume options |
| **API Docs** | API reference for O3Client, O3Dataset, O3CheckpointManager |
| **Settings** | Connect wallet, configure O3 buckets |

### Python: Direct API
```python
from pytorch_o3 import O3Client, O3Dataset, O3CheckpointManager
import torch
from torch.utils.data import DataLoader

# 1. Connect to O3
client = O3Client()  # uses AKAVE_PRIVATE_KEY

# 2. Stream data
object_keys = ["sample_0.pt", "sample_1.pt", 
...]
dataset = O3Dataset(client, "data-bucket", object_keys, 
                      transform=lambda b: torch.load(BytesIO(b)))
loader = DataLoader(dataset, batch_size=32)

# 3. Train model
model = MyModel()
for epoch in range(5):
    for batch_x, batch_y in loader:
        # your training loop
        pass
    
    # 4. Save checkpoint with CID versioning
    ckpt_mgr = O3CheckpointManager(client, "checkpoint-bucket")
    cid = ckpt_mgr.save_checkpoint(
        state_dict=model.state_dict(),
        epoch=epoch,
        optimizer_state=optimizer.state_dict(),
        metrics={"loss": 0.123, "acc": 0.95}
    )
    print(f"Epoch {epoch} → CID: {cid}")
```

### Data Format
Expected object format in O3 buckets:
```python
# Option 1: Dict with {"images": tensor, "labels": tensor}
torch.save({
    "images": torch.randn(1000, 28, 28).uint8(),  # Shape: (N, H, W) or (N, C, H, W)
    "labels": torch.randint(0, 10, (1000,))
}, "bucket_key.pt")

# Option 2: Tuple
torch.save((images_tensor, labels_tensor), "key.pt")
```

### Rate Limiting
Large checkpoint uploads may hit O3/node rate limits (gRPC `RESOURCE_EXHAUSTED`).

**Automatic handling**:
- Retries up to 5 times with 2, 4, 6, 8 minute exponential backoff
- On "file already exists": deletes orphaned key and retries
- Training resumes from latest checkpoint on re-run

**If limits persist**: Wait several minutes and re-run, or reduce checkpoint frequency.

---\n\n## 📚 API Reference\n\n### O3Client — Connect & Stream\n\n```python\nfrom pytorch_o3 import O3Client\n\nclient = O3Client()  # uses AKAVE_PRIVATE_KEY env var\nclient = O3Client(private_key=\"...\", ipc_address=\"connect.akave.ai:5500\")\n```\n\n| Method | Purpose |\n|--------|----------|\n| `list_buckets()` | List all available buckets |\n| `list_objects(bucket, prefix=\"\", limit=1000)` | List objects in a bucket |\n| `get_object_info(bucket, key)` | Get object metadata (size, etc.) |\n| `download_object(bucket, key)` | Download full object → bytes |\n| `download_object_range(bucket, key, start, end)` | Download byte range |\n| `upload_object(bucket, key, data: bytes)` → CID | Upload object, get back CID |\n| `close()` | Close SDK resources |\n\n**Errors**: `O3AuthError` (auth issues), `NotImplementedError` (missing SDK features)\n\n### O3Dataset — Stream to PyTorch\n\n```python\nfrom pytorch_o3 import O3Dataset\nfrom torch.utils.data import DataLoader\n\ndataset = O3Dataset(\n    client=client,\n    bucket_name=\"training-data\",\n    object_keys=[\"sample_0.pt\", \"sample_1.pt\", ...],\n    chunk_size=1024 * 1024,     # 1 MB chunks\n    cache_size=100,              # LRU memory cache\n    transform=None,              # Optional: bytes → sample\n    cache_dir=\"/scratch/o3-cache\" # Optional: persistent disk cache\n)\n\nloader = DataLoader(dataset, batch_size=32, num_workers=4)\nfor batch_x, batch_y in loader:\n    # batch_x, batch_y automatically streamed from O3\n    pass\n```\n\n**Features**:\n- Two-tier caching: LRU memory + SHA256-keyed disk cache\n- Per-worker O3Client (multiprocessing-safe)\n- Automatic chunk fetching on demand\n- Configurable chunk size and cache capacity\n\n**Errors**: `ValueError` (empty keys, bad chunk_size), `RuntimeError` (metadata issues)\n\n### O3CheckpointManager — Versioned Snapshots\n\n```python\nfrom pytorch_o3.checkpoint import O3CheckpointManager\n\nckpt_mgr = O3CheckpointManager(client, bucket_name=\"checkpoints\")\n\n# Save checkpoint with CID versioning\ncid = ckpt_mgr.save_checkpoint(\n    state_dict=model.state_dict(),\n    epoch=5,\n    optimizer_state=optimizer.state_dict(),\n    metrics={\"loss\": 0.123, \"accuracy\": 0.95}\n)\nprint(f\"Checkpoint saved → CID: {cid}\")\n\n# Load latest checkpoint into model\nepoch_to_resume = ckpt_mgr.resume_training(model, optimizer)\nfor epoch in range(epoch_to_resume, total_epochs):\n    # continue training from epoch_to_resume\n    pass\n\n# List all checkpoints\nall_ckpts = ckpt_mgr.list_checkpoints()  # sorted by epoch desc\n\n# Or load specific checkpoint by CID\nckpt_data = ckpt_mgr.load_checkpoint(cid=\"bafy...\")\n```\n\n| Method | Returns | Purpose |\n|--------|---------|----------|\n| `save_checkpoint(...)` | `str` (CID) | Save state, get content-addressed ID |\n| `load_checkpoint(cid=None)` | `dict` | Load by CID or latest if None |\n| `list_checkpoints()` | `list[dict]` | All metadata records (epoch desc) |\n| `get_latest_metadata()` | `dict \\| None` | Newest checkpoint metadata |\n| `get_latest_cid()` | `str \\| None` | Newest checkpoint CID |\n| `resume_training(model, optimizer=None)` | `int` (epoch) | Load latest + return resume epoch |\n\n**Errors**: `RuntimeError` (CID extraction, metadata parsing), upload errors propagate

---

## ⚠️ Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `O3AuthError: AKAVE_PRIVATE_KEY is missing` | Key not exported | Set `AKAVE_PRIVATE_KEY` env var or create .env file |
| `RESOURCE_EXHAUSTED` on upload | Rate limiting on large checkpoints | Auto-retries with backoff; wait 2-8 min + re-run |
| `file already exists` error | Partial upload from crash | Auto-deletes orphaned key and retries |
| Empty `object_keys` → `ValueError` | No dataset objects provided | Pass list of object keys from a bucket |
| `RuntimeError` on CID extraction | Upload successful but CID missing | Check SDK version compatibility |

### Best Practices
✅ **DO**: Use venv, export AKAVE_PRIVATE_KEY, monitor rate limits, use Streamlit dashboard  
❌ **DON'T**: Share private keys, upload huge checkpoints frequently, assume instant O3 uploads

---

## 📦 Project Structure

```
pytorch-o3/
├── src/pytorch_o3/       # Core library
├── examples/             # Training examples
├── tests/                # Unit tests
├── app.py                # Streamlit dashboard
├── demo_training.py      # CLI demo
└── data/
    ├── samples/          # Bundled datasets
    └── checkpoints/      # Local backups
```

---

## ✅ Tests

```bash
python -m pytest tests/ -v  # 25 tests, all passing
```

---

## 🚀 Next Steps

1. **Try CLI**: `python examples/train_mnist.py --o3-data-bucket mnist-data --o3-checkpoint-bucket mnist-ckpt --epochs 2`
2. **Try GUI**: `streamlit run app.py`
3. **Explore API**: Read [API Reference](#api-reference) above
4. **Build custom**: Extend O3Dataset, O3CheckpointManager for your use case

---

## 📄 License & Architecture

- **PyTorch + Akave O3 integration** for decentralized ML  
- Built with **akavesdk** for content-addressed storage  
- CID-based versioning for immutable checkpoint lineage  
- Multiprocessing-safe caching for distributed training

For issues, see [Troubleshooting](#troubleshooting) above.

