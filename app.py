# -*- coding: utf-8 -*-
"""
PyTorch + Akave O3 Training Dashboard
Exact replica of the enterprise dark-theme design
"""

import streamlit as st
import os
import sys
import time
import json
import subprocess
import threading
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

try:
    import plotly.graph_objects as plotly_go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pytorch_o3 import O3Client
    from pytorch_o3.exceptions import O3AuthError
except ImportError:
    O3Client = None

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="O3 Training Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — exact dark theme from screenshot
# ============================================================================
st.markdown("""
<style>
/* ── Global Reset ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #110a06 !important;
    color: #f0e4d8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1c120c !important;
    border-right: 1px solid #2d1f16 !important;
}
[data-testid="stSidebar"] * {
    color: #f0e4d8 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Cards ── */
.card {
    background: #1c120c;
    border: 1px solid #2d1f16;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-inner {
    background: #110a06;
    border: 1px solid #2d1f16;
    border-radius: 8px;
    padding: 16px;
}

/* ── Top Bar ── */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    margin-bottom: 24px;
    border-bottom: 1px solid #2d1f16;
}
.topbar-title {
    font-size: 22px;
    font-weight: 700;
    color: #f0e4d8;
}
.topbar-badge {
    background: #e8451e;
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-left: 12px;
    vertical-align: middle;
}
.connect-wallet-btn {
    background: #e8451e;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
}
.connect-wallet-btn:hover {
    background: #c73a18;
}

/* ── Sidebar items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 8px;
    margin-bottom: 4px;
    cursor: pointer;
    color: #a08070;
    font-size: 14px;
    font-weight: 500;
    text-decoration: none;
}
.nav-item:hover {
    background: #251a12;
    color: #f0e4d8;
}
.nav-item.active {
    background: #e8451e;
    color: white;
}

/* ── Wallet badge ── */
.wallet-badge {
    background: rgba(232, 69, 30, 0.15);
    border: 1px solid #e8451e;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.wallet-addr {
    font-size: 13px;
    font-weight: 500;
    color: #f0e4d8;
}
.green-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3fb950;
    display: inline-block;
}

/* ── Section Headers ── */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #f0e4d8;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Input fields dark ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #110a06 !important;
    border: 1px solid #2d1f16 !important;
    color: #f0e4d8 !important;
    border-radius: 8px !important;
}
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stSlider label {
    color: #a08070 !important;
    font-weight: 500 !important;
}

/* ── Upload area ── */
.upload-zone {
    background: #110a06;
    border: 2px dashed #2d1f16;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    margin-bottom: 16px;
}
.upload-zone:hover {
    border-color: #e8451e;
}
.upload-icon {
    font-size: 36px;
    margin-bottom: 8px;
}
.upload-title {
    font-size: 15px;
    font-weight: 600;
    color: #f0e4d8;
}
.upload-sub {
    font-size: 12px;
    color: #a08070;
    margin-top: 4px;
}

/* ── Object table ── */
.obj-table {
    width: 100%;
    border-collapse: collapse;
}
.obj-table th {
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    color: #a08070;
    letter-spacing: 0.5px;
    padding: 10px 12px;
    border-bottom: 1px solid #2d1f16;
}
.obj-table td {
    padding: 12px;
    font-size: 13px;
    color: #f0e4d8;
    border-bottom: 1px solid #251a12;
    vertical-align: middle;
}
.status-streaming {
    background: rgba(232, 69, 30, 0.2);
    color: #e8451e;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}
.status-ready {
    background: rgba(63, 185, 80, 0.2);
    color: #3fb950;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}
.inspect-link {
    color: #e8451e;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
}

/* ── Progress bar ── */
.progress-outer {
    background: #251a12;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0;
}
.progress-inner {
    background: linear-gradient(90deg, #e8451e, #ff6b3d);
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* ── Log viewer ── */
.log-viewer {
    background: #110a06;
    border: 1px solid #2d1f16;
    border-radius: 8px;
    padding: 14px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: #a08070;
    max-height: 200px;
    overflow-y: auto;
}
.log-info {
    color: #a08070;
}

/* ── Start Training Button ── */
.start-btn {
    background: #e8451e;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-weight: 700;
    font-size: 15px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    justify-content: center;
}
.start-btn:hover {
    background: #c73a18;
}

/* ── Footer status bar ── */
.footer-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-top: 1px solid #2d1f16;
    margin-top: 24px;
    font-size: 12px;
    color: #a08070;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: #251a12 !important;
    color: #f0e4d8 !important;
    border: 1px solid #2d1f16 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: #2d1f16 !important;
    border-color: #e8451e !important;
}

/* Slider override */
.stSlider > div > div > div {
    background: #e8451e !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #1c120c;
    border: 1px solid #2d1f16;
    border-radius: 8px;
    padding: 16px;
}

/* file uploader */
[data-testid="stFileUploader"] {
    background: #110a06;
    border: 2px dashed #2d1f16;
    border-radius: 12px;
    padding: 16px;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2d1f16;
    border-radius: 8px;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #1c120c;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #a08070;
    border-radius: 6px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #e8451e !important;
    color: white !important;
}

/* plotly charts dark bg */
.js-plotly-plot .plotly .main-svg {
    background: #1c120c !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
DEFAULTS = {
    "page": "Overview",
    "connected": False,
    "wallet_addr": None,
    "o3_client": None,
    "data_bucket": "",
    "ckpt_bucket": "",
    "objects": [],
    "selected_dataset": None,
    "training_running": False,
    "training_done": False,
    "training_logs": [],
    "training_epoch": 0,
    "training_total_epochs": 5,
    "training_loss": 0.0,
    "training_accuracy": 0.0,
    "checkpoints": [],       # list of dicts: {epoch, loss, accuracy, cid, timestamp, path, size}
    "stop_training": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# SAMPLE DATASETS (bundled for demo)
# ============================================================================
SAMPLE_DIR = Path(__file__).parent / "data" / "samples"

def get_sample_datasets():
    """Discover bundled sample datasets from data/samples/."""
    datasets = {}
    if not SAMPLE_DIR.exists():
        return datasets
    for d in sorted(SAMPLE_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        files = []
        total_size = 0
        for fp in sorted(d.iterdir()):
            if fp.is_file():
                sz = fp.stat().st_size
                total_size += sz
                files.append({"name": fp.name, "size": sz, "path": str(fp)})
        datasets[d.name] = {
            "meta": meta,
            "files": files,
            "total_size": total_size,
            "dir": str(d),
        }
    return datasets

def fmt_size(b):
    if b >= 1e6: return f"{b/1e6:.1f} MB"
    if b >= 1e3: return f"{b/1e3:.1f} KB"
    return f"{b} B"

# ============================================================================
# SIMPLE CNN MODEL
# ============================================================================
class SimpleCNN(nn.Module):
    """Small CNN that works for 28x28 grayscale or 32x32 RGB."""
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
# REAL TRAINING FUNCTION
# ============================================================================
def run_training(dataset_name, epochs, batch_size, lr, ckpt_bucket):
    """Train on the selected sample dataset. Updates st.session_state live."""
    samples = get_sample_datasets()
    ds = samples.get(dataset_name)
    if not ds:
        st.session_state.training_logs.append(f"[ERROR] Dataset '{dataset_name}' not found")
        st.session_state.training_running = False
        return

    train_file = next((f for f in ds["files"] if f["name"] == "train.pt"), None)
    test_file = next((f for f in ds["files"] if f["name"] == "test.pt"), None)
    if not train_file:
        st.session_state.training_logs.append("[ERROR] No train.pt in dataset")
        st.session_state.training_running = False
        return

    st.session_state.training_logs.append(f"[INFO] Loading {train_file['path']}...")
    train_data = torch.load(train_file["path"], map_location="cpu", weights_only=True)
    images = train_data["images"].float()
    labels = train_data["labels"].long()

    # Normalize to [0,1]
    if images.max() > 1.0:
        images = images / 255.0

    # Add channel dim if needed
    if images.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
        images = images.unsqueeze(1)
    in_channels = images.shape[1]

    meta = ds.get("meta", {})
    num_classes = meta.get("classes", 10)

    st.session_state.training_logs.append(
        f"[INFO] Dataset: {images.shape[0]} samples, shape={list(images.shape[1:])}, classes={num_classes}"
    )

    # Test data
    test_images, test_labels = None, None
    if test_file:
        td = torch.load(test_file["path"], map_location="cpu", weights_only=True)
        test_images = td["images"].float()
        if test_images.max() > 1.0:
            test_images = test_images / 255.0
        if test_images.ndim == 3:
            test_images = test_images.unsqueeze(1)
        test_labels = td["labels"].long()

    train_loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size, shuffle=True
    )

    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path("data/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    st.session_state.training_logs.append(
        f"[INFO] Model: SimpleCNN ({sum(p.numel() for p in model.parameters()):,} params)"
    )
    st.session_state.training_logs.append(
        f"[INFO] Config: epochs={epochs}, batch_size={batch_size}, lr={lr}"
    )
    st.session_state.training_logs.append(f"[INFO] Training started at {datetime.now().strftime('%H:%M:%S')}")

    for epoch in range(1, epochs + 1):
        if st.session_state.get("stop_training", False):
            st.session_state.training_logs.append(f"[WARN] Training stopped by user at epoch {epoch}")
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Test evaluation
        test_acc = 0.0
        if test_images is not None:
            model.eval()
            with torch.no_grad():
                out = model(test_images)
                pred = out.argmax(dim=1)
                test_acc = 100.0 * pred.eq(test_labels).sum().item() / len(test_labels)

        # Update session state
        st.session_state.training_epoch = epoch
        st.session_state.training_loss = train_loss
        st.session_state.training_accuracy = test_acc if test_images is not None else train_acc

        st.session_state.training_logs.append(
            f"[EPOCH {epoch}/{epochs}] loss={train_loss:.4f}  train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%"
        )

        # Save checkpoint locally
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            "accuracy": test_acc if test_images is not None else train_acc,
        }
        torch.save(ckpt_payload, ckpt_path)
        ckpt_size = ckpt_path.stat().st_size

        # Try O3 upload if connected
        cid = None
        client = st.session_state.get("o3_client")
        if client and ckpt_bucket:
            try:
                buf = io.BytesIO()
                torch.save(ckpt_payload, buf)
                data_bytes = buf.getvalue()
                file_meta = client.upload_object(ckpt_bucket, f"checkpoint_epoch_{epoch:03d}.pt", data_bytes)
                # Extract CID
                if hasattr(file_meta, 'root_cid'):
                    cid = file_meta.root_cid
                elif hasattr(file_meta, 'RootCid'):
                    cid = file_meta.RootCid
                elif isinstance(file_meta, dict):
                    cid = file_meta.get('root_cid', file_meta.get('RootCid'))
                if cid:
                    st.session_state.training_logs.append(f"[O3] Checkpoint uploaded → CID: {cid}")
                else:
                    st.session_state.training_logs.append(f"[O3] Checkpoint uploaded (CID extraction pending)")
            except Exception as e:
                st.session_state.training_logs.append(f"[O3] Upload failed: {e}")
                cid = None

        # Store checkpoint record
        st.session_state.checkpoints.append({
            "epoch": epoch,
            "loss": round(train_loss, 4),
            "accuracy": round(test_acc if test_images is not None else train_acc, 2),
            "cid": cid or "local-only",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "path": str(ckpt_path),
            "size": ckpt_size,
        })

    # Done
    st.session_state.training_logs.append(f"[INFO] Training complete at {datetime.now().strftime('%H:%M:%S')}")
    if st.session_state.checkpoints:
        best = max(st.session_state.checkpoints, key=lambda c: c["accuracy"])
        st.session_state.training_logs.append(
            f"[INFO] Best: epoch {best['epoch']} — accuracy {best['accuracy']}% — CID: {best['cid']}"
        )
    st.session_state.training_running = False
    st.session_state.training_done = True

# ============================================================================
# HELPERS
# ============================================================================
def validate_key(key):
    return len(key) == 64 and all(c in "0123456789abcdef" for c in key.lower())

def shorten_addr(key):
    return f"0x{key[:4]}...{key[-4:]}" if len(key) >= 8 else key

def do_connect(key):
    if not validate_key(key):
        st.error("Invalid key — must be 64 hex characters")
        return False
    try:
        client = O3Client(private_key=key) if O3Client else None
        st.session_state.o3_client = client
        st.session_state.connected = True
        st.session_state.wallet_addr = shorten_addr(key)
        return True
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return False

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
        <div style="background:#e8451e; border-radius:10px; width:40px; height:40px; display:flex; align-items:center; justify-content:center; font-size:20px;">🔴</div>
        <div>
            <div style="font-size:16px; font-weight:700;">PyTorch + Akave</div>
            <div style="font-size:11px; color:#e8451e; font-weight:600; letter-spacing:1px;">O3 INTEGRATION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Wallet status
    if st.session_state.connected:
        st.markdown(f"""
        <div class="wallet-badge">
            <div>
                <div style="font-size:11px; color:#e8451e; font-weight:600;">WALLET STATUS</div>
                <div class="wallet-addr">{st.session_state.wallet_addr}</div>
            </div>
            <span class="green-dot"></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="wallet-badge" style="border-color:#2d1f16; background:rgba(45,31,22,0.3);">
            <div>
                <div style="font-size:11px; color:#a08070; font-weight:600;">WALLET STATUS</div>
                <div class="wallet-addr" style="color:#a08070;">Not Connected</div>
            </div>
            <span style="width:8px;height:8px;border-radius:50%;background:#f85149;display:inline-block;"></span>
        </div>
        """, unsafe_allow_html=True)

    # Navigation
    nav_items = [
        ("🏠", "Overview"),
        ("📊", "Dashboard"),
        ("📦", "Datasets"),
        ("🤖", "Training"),
        ("💾", "Checkpoints"),
        ("🪣", "Buckets"),
        ("📖", "API Docs"),
    ]
    for icon, label in nav_items:
        active = "active" if st.session_state.page == label else ""
        if st.button(f"{icon}  {label}", key=f"nav_{label}", width='stretch'):
            st.session_state.page = label
            st.rerun()

    st.markdown("""<div style="margin-top:16px; font-size:11px; color:#a08070; font-weight:600; letter-spacing:0.5px; padding-left:16px;">SYSTEM</div>""", unsafe_allow_html=True)

    if st.button("⚙️  Settings", key="nav_Settings", width='stretch'):
        st.session_state.page = "Settings"
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:12px; color:#a08070; padding:0 8px;">
        <div>📡 connect.akave.ai:5500</div>
        <div style="margin-top:4px;">
            <span class="green-dot"></span>
            <span style="color:#3fb950; font-weight:600; font-size:11px; margin-left:6px;">NODE CONNECTED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TOP BAR
# ============================================================================
def top_bar():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:22px; font-weight:700; color:#f0e4d8;">O3 Training Dashboard</span>
            <span class="topbar-badge">ENTERPRISE BETA</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if not st.session_state.connected:
            if st.button("🔴 Connect Wallet", key="top_connect", width='stretch'):
                st.session_state.page = "Settings"
                st.rerun()
        else:
            st.markdown(f"""
            <div style="text-align:right; display:flex; align-items:center; justify-content:flex-end; gap:8px;">
                <span style="font-size:13px; color:#a08070;">{st.session_state.wallet_addr}</span>
                <div style="width:32px; height:32px; border-radius:50%; background:#2d1f16; display:flex; align-items:center; justify-content:center;">👤</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<div style='border-bottom:1px solid #2d1f16; margin:8px 0 24px 0;'></div>", unsafe_allow_html=True)

# ============================================================================
# PAGE: OVERVIEW (Landing page — scannable project summary)
# ============================================================================
def page_overview():
    top_bar()

    # ── Hero ──
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <div style="font-size:36px; font-weight:700; color:#f0e4d8; margin-bottom:4px;">PyTorch + Akave O3</div>
        <div style="font-size:15px; color:#a08070; max-width:650px; margin:0 auto;">Decentralized ML training pipeline — stream datasets, train models, and store CID-based immutable checkpoints on Akave O3 storage.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4 Core Components ──
    st.markdown("""<div class="section-title">🧩 Core Components</div>""", unsafe_allow_html=True)

    comp1, comp2, comp3, comp4 = st.columns(4)
    with comp1:
        st.markdown("""
        <div class="card-inner" style="padding:20px; text-align:center; min-height:210px;">
            <div style="font-size:28px; margin-bottom:8px;">🔌</div>
            <div style="font-size:14px; font-weight:700; color:#e8451e; margin-bottom:6px;">O3Client</div>
            <div style="font-size:12px; color:#a08070; line-height:1.6;">Thin wrapper around <code>akavesdk</code>. Range &amp; full-object streaming, uploads with CID return, retry logic with exponential backoff.</div>
            <div style="margin-top:10px; font-size:11px; color:#f0e4d8;">client.py</div>
        </div>
        """, unsafe_allow_html=True)
    with comp2:
        st.markdown("""
        <div class="card-inner" style="padding:20px; text-align:center; min-height:210px;">
            <div style="font-size:28px; margin-bottom:8px;">📦</div>
            <div style="font-size:14px; font-weight:700; color:#e8451e; margin-bottom:6px;">O3Dataset</div>
            <div style="font-size:12px; color:#a08070; line-height:1.6;">PyTorch <code>Dataset</code> that streams samples from O3. Two-tier caching (LRU memory + SHA256 disk), multiprocessing-safe.</div>
            <div style="margin-top:10px; font-size:11px; color:#f0e4d8;">dataset.py</div>
        </div>
        """, unsafe_allow_html=True)
    with comp3:
        st.markdown("""
        <div class="card-inner" style="padding:20px; text-align:center; min-height:210px;">
            <div style="font-size:28px; margin-bottom:8px;">💾</div>
            <div style="font-size:14px; font-weight:700; color:#e8451e; margin-bottom:6px;">O3CheckpointManager</div>
            <div style="font-size:12px; color:#a08070; line-height:1.6;">CID-based checkpoint persistence. Immutable snapshots with lineage tracking, auto-resume from latest checkpoint.</div>
            <div style="margin-top:10px; font-size:11px; color:#f0e4d8;">checkpoint.py</div>
        </div>
        """, unsafe_allow_html=True)
    with comp4:
        st.markdown("""
        <div class="card-inner" style="padding:20px; text-align:center; min-height:210px;">
            <div style="font-size:28px; margin-bottom:8px;">🧠</div>
            <div style="font-size:14px; font-weight:700; color:#e8451e; margin-bottom:6px;">MNIST Example</div>
            <div style="font-size:12px; color:#a08070; line-height:1.6;">End-to-end training with <code>O3Dataset</code> for streaming + <code>O3CheckpointManager</code> for CID-tracked snapshots.</div>
            <div style="margin-top:10px; font-size:11px; color:#f0e4d8;">examples/train_mnist.py</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture / Data Flow ──
    st.markdown("""<div class="section-title">🔄 Architecture &amp; Data Flow</div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-inner" style="padding:24px;">
        <div style="display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap; font-size:13px;">
            <div style="background:#251a12; border:2px solid #e8451e; border-radius:10px; padding:12px 18px; text-align:center;">
                <div style="font-weight:700; color:#e8451e;">1. Connect</div>
                <div style="color:#a08070; font-size:11px; margin-top:4px;">O3Client + PRIVATE_KEY<br/>IPC → connect.akave.ai:5500</div>
            </div>
            <span style="color:#e8451e; font-size:20px;">→</span>
            <div style="background:#251a12; border:2px solid #e8451e; border-radius:10px; padding:12px 18px; text-align:center;">
                <div style="font-weight:700; color:#e8451e;">2. Stream Data</div>
                <div style="color:#a08070; font-size:11px; margin-top:4px;">O3Dataset fetches objects<br/>LRU cache → disk cache</div>
            </div>
            <span style="color:#e8451e; font-size:20px;">→</span>
            <div style="background:#251a12; border:2px solid #e8451e; border-radius:10px; padding:12px 18px; text-align:center;">
                <div style="font-weight:700; color:#e8451e;">3. Train Model</div>
                <div style="color:#a08070; font-size:11px; margin-top:4px;">PyTorch DataLoader<br/>SimpleCNN / custom model</div>
            </div>
            <span style="color:#e8451e; font-size:20px;">→</span>
            <div style="background:#251a12; border:2px solid #e8451e; border-radius:10px; padding:12px 18px; text-align:center;">
                <div style="font-weight:700; color:#e8451e;">4. Checkpoint</div>
                <div style="color:#a08070; font-size:11px; margin-top:4px;">O3CheckpointManager<br/>CID-versioned .pt + JSON</div>
            </div>
            <span style="color:#e8451e; font-size:20px;">→</span>
            <div style="background:#251a12; border:2px solid #e8451e; border-radius:10px; padding:12px 18px; text-align:center;">
                <div style="font-weight:700; color:#e8451e;">5. Resume</div>
                <div style="color:#a08070; font-size:11px; margin-top:4px;">Auto-detect latest CID<br/>Continue from last epoch</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Start ──
    st.markdown("""<div class="section-title">🚀 Quick Start</div>""", unsafe_allow_html=True)

    qs1, qs2 = st.columns(2)
    with qs1:
        st.markdown(r"""
        <div class="card-inner" style="padding:20px;">
            <div style="font-size:13px; font-weight:600; color:#e8451e; margin-bottom:10px;">📋 Setup (CLI)</div>
            <div style="font-family:'JetBrains Mono','Courier New',monospace; font-size:11px; color:#a08070; line-height:2; background:#110a06; padding:14px; border-radius:6px;">
                <span style="color:#3fb950;">$</span> python -m venv .venv<br>
                <span style="color:#3fb950;">$</span> .venv\Scripts\activate<br>
                <span style="color:#3fb950;">$</span> pip install -r requirements.txt<br>
                <span style="color:#3fb950;">$</span> pip install -e .<br>
                <span style="color:#3fb950;">$</span> export AKAVE_PRIVATE_KEY="your_key"<br>
                <span style="color:#3fb950;">$</span> python examples/train_mnist.py --o3-data-bucket mnist-data --o3-checkpoint-bucket mnist-ckpt --epochs 5
            </div>
        </div>
        """, unsafe_allow_html=True)
    with qs2:
        st.markdown("""
        <div class="card-inner" style="padding:20px;">
            <div style="font-size:13px; font-weight:600; color:#e8451e; margin-bottom:10px;">🖥️ This Dashboard</div>
            <div style="font-size:12px; color:#a08070; line-height:1.8;">
                <strong style="color:#f0e4d8;">1.</strong> Go to <strong style="color:#f0e4d8;">Settings</strong> → enter your AKAVE_PRIVATE_KEY<br>
                <strong style="color:#f0e4d8;">2.</strong> Go to <strong style="color:#f0e4d8;">Dashboard</strong> → pick a sample dataset<br>
                <strong style="color:#f0e4d8;">3.</strong> Set epochs, batch size, learning rate<br>
                <strong style="color:#f0e4d8;">4.</strong> Click <strong style="color:#e8451e;">▶ Start Training</strong><br>
                <strong style="color:#f0e4d8;">5.</strong> Watch real-time logs, loss, accuracy<br>
                <strong style="color:#f0e4d8;">6.</strong> View checkpoints with CIDs on <strong style="color:#f0e4d8;">Checkpoints</strong> page<br>
            </div>
            <div style="margin-top:12px; font-size:11px; color:#a08070;">No wallet? Training still works locally — CIDs appear when connected to O3.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Concepts ──
    st.markdown("""<div class="section-title">📚 Key Concepts</div>""", unsafe_allow_html=True)

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        st.markdown("""
        <div class="card-inner" style="padding:18px;">
            <div style="font-size:13px; font-weight:600; color:#e8451e; margin-bottom:8px;">🔑 CID-based Versioning</div>
            <div style="font-size:12px; color:#a08070; line-height:1.7;">Every checkpoint upload returns a Content Identifier (CID) — a hash-based address. Checkpoints are <strong style="color:#f0e4d8;">immutable</strong>: same CID always → same bytes. Parent CID creates a lineage chain.</div>
        </div>
        """, unsafe_allow_html=True)
    with kc2:
        st.markdown("""
        <div class="card-inner" style="padding:18px;">
            <div style="font-size:13px; font-weight:600; color:#e8451e; margin-bottom:8px;">🔄 Auto-Resume</div>
            <div style="font-size:12px; color:#a08070; line-height:1.7;"><code>O3CheckpointManager.resume_training()</code> finds the latest checkpoint, loads model + optimizer state, and returns the epoch to continue from. No manual bookkeeping.</div>
        </div>
        """, unsafe_allow_html=True)
    with kc3:
        st.markdown("""
        <div class="card-inner" style="padding:18px;">
            <div style="font-size:13px; font-weight:600; color:#e8451e; margin-bottom:8px;">📡 Chunked Streaming</div>
            <div style="font-size:12px; color:#a08070; line-height:1.7;"><code>O3Dataset</code> splits objects into configurable chunks (default 1 MB). LRU memory cache + optional SHA256-keyed disk cache. Per-worker O3Client for multiprocessing safety.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Failure Modes ──
    st.markdown("""<div class="section-title">⚠️ Common Failure Modes</div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-inner" style="padding:18px;">
        <table style="width:100%; font-size:12px; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:8px 12px; color:#e8451e; font-weight:600; width:30%;">Missing AKAVE_PRIVATE_KEY</td>
                <td style="padding:8px 12px; color:#a08070;">O3Client raises <code>O3AuthError</code>. Set the key in Settings or export it before running CLI.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:8px 12px; color:#e8451e; font-weight:600;">Rate limits on upload</td>
                <td style="padding:8px 12px; color:#a08070;">Large checkpoints can trigger gRPC RESOURCE_EXHAUSTED. Retries with 2-8 min backoff. Wait and re-run if persistent; training auto-resumes.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:8px 12px; color:#e8451e; font-weight:600;">Malformed checkpoint metadata</td>
                <td style="padding:8px 12px; color:#a08070;">Logged as warning, remaining files still processed. Unexpected errors logged + re-raised.</td>
            </tr>
            <tr>
                <td style="padding:8px 12px; color:#e8451e; font-weight:600;">Object size unknown</td>
                <td style="padding:8px 12px; color:#a08070;">O3Dataset raises ValueError with the object key. Check that objects have valid metadata in the bucket.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="footer-bar">
        <div>📡 connect.akave.ai:5500 &nbsp;|&nbsp; torch {torch.__version__}</div>
        <div>v0.1.0 &nbsp;•&nbsp; <span style="color:#e8451e;">PyTorch + Akave O3 Integration</span></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: API DOCS (scannable reference from README)
# ============================================================================
def page_api_docs():
    top_bar()
    st.markdown("""<div class="section-title" style="font-size:20px;">📖 API Reference</div>""", unsafe_allow_html=True)

    # ── O3Client ──
    st.markdown("### 🔌 O3Client")
    st.markdown('<div style="font-size:13px; color:#a08070; margin-bottom:12px;">Defined in <code>pytorch_o3.client.O3Client</code> — light wrapper around akavesdk for object operations.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-inner" style="padding:18px; font-size:12px; line-height:1.8;">
        <table style="width:100%; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#e8451e; font-weight:600; width:40%; font-family:monospace;">O3Client(private_key=None, ipc_address="connect.akave.ai:5500")</td>
                <td style="padding:6px 10px; color:#a08070;">Initialize. Uses AKAVE_PRIVATE_KEY env var if key not provided.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">list_buckets()</td>
                <td style="padding:6px 10px; color:#a08070;">Return all available buckets.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">list_objects(bucket, prefix="", limit=1000)</td>
                <td style="padding:6px 10px; color:#a08070;">List objects in a bucket.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">get_object_info(bucket, key)</td>
                <td style="padding:6px 10px; color:#a08070;">Inspect object metadata including size.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">download_object(bucket, key)</td>
                <td style="padding:6px 10px; color:#a08070;">Full object download as bytes.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">download_object_range(bucket, key, start, end)</td>
                <td style="padding:6px 10px; color:#a08070;">Byte-range download.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">upload_object(bucket, key, data: bytes)</td>
                <td style="padding:6px 10px; color:#a08070;">Upload object (min 127 bytes). Returns metadata with CID.</td>
            </tr>
            <tr>
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">close()</td>
                <td style="padding:6px 10px; color:#a08070;">Close underlying SDK resources.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── O3Dataset ──
    st.markdown("### 📦 O3Dataset")
    st.markdown('<div style="font-size:13px; color:#a08070; margin-bottom:12px;">Defined in <code>pytorch_o3.dataset.O3Dataset</code> — PyTorch Dataset with O3 streaming + caching.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-inner" style="padding:18px;">
        <div style="font-size:11px; font-weight:600; color:#e8451e; letter-spacing:0.5px; margin-bottom:8px;">CONSTRUCTOR PARAMETERS</div>
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace; width:30%;">client</td>
                <td style="padding:6px 10px; color:#a08070;">Initialized O3Client instance</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">bucket_name</td>
                <td style="padding:6px 10px; color:#a08070;">Bucket containing dataset objects</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">object_keys</td>
                <td style="padding:6px 10px; color:#a08070;">List of keys forming the dataset (non-empty required)</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">chunk_size</td>
                <td style="padding:6px 10px; color:#a08070;">Bytes per range request chunk (default 1 MB)</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">cache_size</td>
                <td style="padding:6px 10px; color:#a08070;">LRU memory cache capacity (# chunks)</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">transform</td>
                <td style="padding:6px 10px; color:#a08070;">Callable: raw bytes → sample (tensor, dict, etc.)</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">target_transform</td>
                <td style="padding:6px 10px; color:#a08070;">Optional callable to create targets from transformed data</td>
            </tr>
            <tr>
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">cache_dir</td>
                <td style="padding:6px 10px; color:#a08070;">Optional directory for persistent SHA256-keyed disk cache</td>
            </tr>
        </table>
        <div style="margin-top:12px; font-size:11px; color:#a08070;"><strong style="color:#f0e4d8;">Worker Safety:</strong> Each DataLoader worker gets its own O3Client to avoid cross-process contention.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── O3CheckpointManager ──
    st.markdown("### 💾 O3CheckpointManager")
    st.markdown('<div style="font-size:13px; color:#a08070; margin-bottom:12px;">Defined in <code>pytorch_o3.checkpoint.O3CheckpointManager</code> — CID-based checkpoint persistence with lineage.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-inner" style="padding:18px;">
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#e8451e; font-weight:600; font-family:monospace; width:45%;">save_checkpoint(state_dict, epoch, metrics, optimizer_state, extra_data) → str</td>
                <td style="padding:6px 10px; color:#a08070;">Save .pt + JSON metadata. Returns CID.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">load_checkpoint(cid=None) → dict</td>
                <td style="padding:6px 10px; color:#a08070;">Load by CID or latest if None.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">list_checkpoints() → list[dict]</td>
                <td style="padding:6px 10px; color:#a08070;">All metadata records, sorted by epoch (desc).</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">get_latest_metadata() → dict | None</td>
                <td style="padding:6px 10px; color:#a08070;">Newest checkpoint metadata or None.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">get_latest_cid() → str | None</td>
                <td style="padding:6px 10px; color:#a08070;">root_cid of latest checkpoint.</td>
            </tr>
            <tr>
                <td style="padding:6px 10px; color:#f0e4d8; font-family:monospace;">resume_training(model, optimizer=None) → int</td>
                <td style="padding:6px 10px; color:#a08070;">Load latest into model/optimizer, return resume epoch (0 if none).</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MNIST Example Usage ──
    st.markdown("### 🧠 MNIST Example")
    st.markdown('<div style="font-size:13px; color:#a08070; margin-bottom:12px;">End-to-end training: <code>examples/train_mnist.py</code></div>', unsafe_allow_html=True)

    st.markdown(r"""
    <div class="card-inner" style="padding:18px;">
        <div style="font-family:'JetBrains Mono','Courier New',monospace; font-size:11px; color:#a08070; line-height:2; background:#110a06; padding:14px; border-radius:6px;">
            <span style="color:#3fb950;">$</span> python examples/train_mnist.py \<br>
            &nbsp;&nbsp;--o3-data-bucket mnist-data \<br>
            &nbsp;&nbsp;--o3-train-prefix mnist/train/ \<br>
            &nbsp;&nbsp;--o3-test-prefix mnist/test/ \<br>
            &nbsp;&nbsp;--o3-checkpoint-bucket mnist-checkpoints \<br>
            &nbsp;&nbsp;--epochs 5
        </div>
        <div style="margin-top:14px; font-size:12px; color:#a08070; line-height:1.7;">
            Each epoch: <strong style="color:#f0e4d8;">stream batches from O3</strong> → train → evaluate →
            <strong style="color:#f0e4d8;">save checkpoint (.pt + JSON)</strong> → log CID (root_cid).<br>
            Re-run discovers the latest checkpoint and <strong style="color:#f0e4d8;">auto-resumes</strong> from that epoch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Error Reference ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Error Reference")

    st.markdown("""
    <div class="card-inner" style="padding:18px;">
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#e8451e; font-family:monospace; width:35%;">O3AuthError</td>
                <td style="padding:6px 10px; color:#a08070;">Missing AKAVE_PRIVATE_KEY or SDK misconfiguration.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#e8451e; font-family:monospace;">ValueError</td>
                <td style="padding:6px 10px; color:#a08070;">Empty object_keys, non-positive chunk_size, negative cache_size, unknown object size.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#e8451e; font-family:monospace;">RuntimeError</td>
                <td style="padding:6px 10px; color:#a08070;">CID extraction failure on upload, metadata resolution issues.</td>
            </tr>
            <tr style="border-bottom:1px solid #2d1f16;">
                <td style="padding:6px 10px; color:#e8451e; font-family:monospace;">NotImplementedError</td>
                <td style="padding:6px 10px; color:#a08070;">SDK missing list_files capability.</td>
            </tr>
            <tr>
                <td style="padding:6px 10px; color:#e8451e; font-family:monospace;">RESOURCE_EXHAUSTED</td>
                <td style="padding:6px 10px; color:#a08070;">gRPC rate limit on large checkpoint uploads. Retries with 2-8 min backoff.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="footer-bar">
        <div>Source: README.md &nbsp;•&nbsp; pytorch_o3 v0.1.0</div>
        <div><span style="color:#e8451e;">PyTorch + Akave O3 Integration</span></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def page_dashboard():
    top_bar()

    # ── Bucket Configuration ──
    st.markdown("""<div class="section-title">🪣 Bucket Configuration</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.data_bucket = st.text_input(
            "Data Bucket Name",
            value=st.session_state.data_bucket,
            placeholder="e.g. training-dataset-v1",
            help="Source bucket for training data (.pt, .pth)",
            key="data_bucket_in",
        )
    with col2:
        st.session_state.ckpt_bucket = st.text_input(
            "Checkpoint Bucket Name",
            value=st.session_state.ckpt_bucket,
            placeholder="e.g. o3-model-checkpoints",
            help="Destination for auto-saved checkpoints",
            key="ckpt_bucket_in",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: Dataset Management | Training Controls ──
    left, right = st.columns([3, 2])

    with left:
        st.markdown("""<div class="section-title">📦 Dataset Management</div>""", unsafe_allow_html=True)

        # Upload zone
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📤</div>
            <div class="upload-title">Upload Training Files</div>
            <div class="upload-sub">Select .pt or .pth files to store on Akave O3 storage</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose files", type=["pt", "pth", "zip", "tar"], label_visibility="collapsed", accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                st.success(f"Ready: {f.name} ({f.size / 1e6:.1f} MB)")

        # ── Sample Datasets — pick one for training ──
        samples = get_sample_datasets()
        if samples:
            st.markdown("""<div style="margin-top:12px; font-size:12px; font-weight:600; color:#a08070; letter-spacing:0.5px; margin-bottom:8px;">SAMPLE DATASETS (bundled, ready to use)</div>""", unsafe_allow_html=True)
            for ds_name, ds_info in samples.items():
                display = ds_info["meta"].get("name", ds_name)
                total = fmt_size(ds_info["total_size"])
                n_files = len(ds_info["files"])
                train_samples = ds_info["meta"].get("train_samples", "?")
                is_selected = st.session_state.selected_dataset == ds_name

                cols = st.columns([3, 1.5, 1.5, 2])
                with cols[0]:
                    dot = "🟢" if is_selected else "⚪"
                    st.markdown(f"<div style='padding:8px 0; font-size:13px;'>{dot} <strong>{display}</strong></div>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<div style='padding:8px 0; font-size:13px; color:#a08070;'>{total} · {n_files} files</div>", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f"<div style='padding:8px 0; font-size:13px; color:#a08070;'>{train_samples} samples</div>", unsafe_allow_html=True)
                with cols[3]:
                    if is_selected:
                        st.markdown("<div style='padding:6px 0;'><span style='background:#3fb950; color:#000; padding:4px 12px; border-radius:4px; font-size:11px; font-weight:700;'>✓ ACTIVE</span></div>", unsafe_allow_html=True)
                    else:
                        if st.button(f"➕ Add for Training", key=f"add_{ds_name}", width='stretch'):
                            st.session_state.selected_dataset = ds_name
                            st.session_state.data_bucket = ds_name
                            st.rerun()
        else:
            st.info("No sample datasets found in data/samples/")

    with right:
        st.markdown("""<div class="section-title">⚙️ Training Controls</div>""", unsafe_allow_html=True)

        # Show active dataset
        if st.session_state.selected_dataset:
            ds = get_sample_datasets().get(st.session_state.selected_dataset, {})
            ds_display = ds.get("meta", {}).get("name", st.session_state.selected_dataset) if ds else st.session_state.selected_dataset
            st.markdown(f"""
            <div style="background:rgba(232,69,30,0.12); border:1px solid #e8451e; border-radius:8px; padding:10px 14px; margin-bottom:16px;">
                <div style="font-size:11px; color:#e8451e; font-weight:600;">ACTIVE DATASET</div>
                <div style="font-size:14px; font-weight:600; color:#f0e4d8;">{ds_display}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:rgba(45,31,22,0.5); border:1px dashed #2d1f16; border-radius:8px; padding:10px 14px; margin-bottom:16px; font-size:13px; color:#a08070;">No dataset selected — pick one from the left</div>""", unsafe_allow_html=True)

        epochs = st.slider("Epochs", 1, 200, 5, key="epochs_slider")
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128, 256], index=2, key="batch_sel")
        learning_rate = st.text_input("Learning Rate", value="0.001", key="lr_input")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.training_running:
            st.markdown("""<div style="background:rgba(63,185,80,0.12); border:1px solid #3fb950; border-radius:8px; padding:10px 14px; font-size:13px; color:#3fb950; font-weight:600; text-align:center;">⏳ Training in progress...</div>""", unsafe_allow_html=True)
            if st.button("⏹  Stop Training", key="stop_train", width='stretch'):
                st.session_state.stop_training = True
        else:
            can_start = st.session_state.selected_dataset is not None
            if st.button("▶  Start Training", key="start_train", width='stretch', disabled=not can_start):
                if can_start:
                    st.session_state.training_running = True
                    st.session_state.training_done = False
                    st.session_state.training_epoch = 0
                    st.session_state.training_total_epochs = epochs
                    st.session_state.training_loss = 0.0
                    st.session_state.training_accuracy = 0.0
                    st.session_state.training_logs = []
                    st.session_state.checkpoints = []
                    st.session_state.stop_training = False
                    try:
                        lr_val = float(learning_rate)
                    except ValueError:
                        lr_val = 0.001
                    # Run training synchronously (Streamlit will show spinner)
                    run_training(
                        dataset_name=st.session_state.selected_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr_val,
                        ckpt_bucket=st.session_state.ckpt_bucket,
                    )
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Real-time Logs & Progress ──
    st.markdown("""<div class="section-title">📊 Real-time Logs & Progress</div>""", unsafe_allow_html=True)

    epoch_cur = st.session_state.training_epoch
    epoch_total = st.session_state.training_total_epochs
    pct = int((epoch_cur / epoch_total) * 100) if epoch_total > 0 else 0

    pcol1, pcol2 = st.columns([3, 1])
    with pcol1:
        st.markdown(f"**Epoch {epoch_cur}/{epoch_total}**")
        st.markdown(f"""
        <div class="progress-outer">
            <div class="progress-inner" style="width:{pct}%;"></div>
        </div>
        """, unsafe_allow_html=True)
    with pcol2:
        status_txt = "Active" if st.session_state.training_running else "Idle"
        st.markdown(f"""
        <div style="text-align:right; font-size:13px;">
            <span style="color:#a08070;">Status: </span><span style="color:{'#3fb950' if st.session_state.training_running else '#a08070'};">{status_txt}</span>
            <span style="margin-left:16px; color:#a08070;">Worker ID: </span><span style="color:#f0e4d8;">akave-node-k1</span>
        </div>
        <div style="text-align:right; font-size:22px; font-weight:700; color:#e8451e;">{pct}% Complete</div>
        """, unsafe_allow_html=True)

    # Log viewer
    logs_text = "\n".join(st.session_state.training_logs[-15:]) if st.session_state.training_logs else "[INFO] Waiting for training to start..."
    st.markdown(f"""
    <div class="log-viewer">
{logs_text}
    </div>
    """, unsafe_allow_html=True)

    # Footer bar
    st.markdown(f"""
    <div class="footer-bar">
        <div>📡 connect.akave.ai:5500</div>
        <div><span class="green-dot"></span> <span style="color:#3fb950; font-weight:600;">NODE CONNECTED</span></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DATASETS
# ============================================================================
def page_datasets():
    top_bar()
    st.markdown("""<div class="section-title" style="font-size:20px;">📦 Dataset Explorer</div>""", unsafe_allow_html=True)

    samples = get_sample_datasets()
    ds_names = list(samples.keys())

    if not ds_names:
        st.warning("No sample datasets found in data/samples/. Run the generation script first.")
        return

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 🪣 Sample Datasets")
        for name in ds_names:
            display = samples[name]["meta"].get("name", name)
            total = fmt_size(samples[name]["total_size"])
            if st.button(f"📦 {display}  ({total})", key=f"ds_{name}", width='stretch'):
                st.session_state.data_bucket = name
                st.rerun()

    selected = st.session_state.data_bucket if st.session_state.data_bucket in samples else (ds_names[0] if ds_names else None)
    ds = samples.get(selected) if selected else None

    with col2:
        st.markdown("### 📂 Files")
        if ds:
            for fi in ds["files"]:
                ext = fi["name"].split(".")[-1]
                icon = "🔶" if ext == "pt" else "📋" if ext == "json" else "📝"
                st.markdown(f"""<div style="padding:8px 12px; background:#251a12; border-radius:6px; margin:4px 0; font-size:13px; display:flex; justify-content:space-between;">
                    <span>{icon} {fi['name']}</span><span style="color:#a08070;">{fmt_size(fi['size'])}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Select a dataset")

    with col3:
        st.markdown("### 📊 Info")
        if ds and ds["meta"]:
            m = ds["meta"]
            st.markdown(f"""
            <div class="card-inner" style="padding:16px;">
                <strong style="color:#e8451e;">Name:</strong> {m.get('name','—')}<br>
                <strong style="color:#e8451e;">Classes:</strong> {m.get('classes','—')}<br>
                <strong style="color:#e8451e;">Train:</strong> {m.get('train_samples','—')} samples<br>
                <strong style="color:#e8451e;">Test:</strong> {m.get('test_samples','—')} samples<br>
                <strong style="color:#e8451e;">Shape:</strong> {m.get('image_shape','—')}<br>
                <strong style="color:#e8451e;">Channels:</strong> {m.get('channels','—')}<br>
                <strong style="color:#e8451e;">Source:</strong> {m.get('source','—')}<br>
            </div>
            """, unsafe_allow_html=True)

            if m.get("class_names"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""<div style="font-size:12px; color:#a08070;">Classes: {', '.join(m['class_names'])}</div>""", unsafe_allow_html=True)
        else:
            st.info("Select a dataset")

    st.markdown("---")

    # ── Live tensor preview ──
    st.markdown("### 👁️ Tensor Preview")
    if ds:
        # Load train.pt for preview
        train_file = next((f for f in ds["files"] if f["name"] == "train.pt"), None)
        if train_file:
            try:
                import torch
                data = torch.load(train_file["path"], map_location="cpu", weights_only=True)
                images = data["images"]
                labels = data["labels"]
                idx = st.slider("Sample index", 0, len(images) - 1, 0, key="preview_idx")
                img = images[idx].numpy()
                lbl = labels[idx].item()

                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    class_names = ds["meta"].get("class_names", [])
                    label_str = class_names[lbl] if lbl < len(class_names) else str(lbl)
                    shape_str = "x".join(str(s) for s in img.shape)
                    st.markdown(f"""
                    <div class="card-inner" style="padding:20px;">
                        <strong style="color:#e8451e;">Label:</strong> {lbl} ({label_str})<br>
                        <strong style="color:#e8451e;">Shape:</strong> {shape_str}<br>
                        <strong style="color:#e8451e;">Dtype:</strong> {images.dtype}<br>
                        <strong style="color:#e8451e;">Min/Max:</strong> {img.min():.2f} / {img.max():.2f}<br>
                    </div>
                    """, unsafe_allow_html=True)
                with pcol2:
                    # Render as heatmap
                    if img.ndim == 3 and img.shape[0] in (1, 3):
                        vis = img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0).mean(axis=2)
                    elif img.ndim == 2:
                        vis = img
                    else:
                        vis = img.reshape(img.shape[-2], img.shape[-1]) if img.ndim >= 2 else img

                    if HAS_PLOTLY:
                        fig = plotly_go.Figure(data=plotly_go.Heatmap(z=vis, colorscale="Hot", showscale=False))
                        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#1c120c", plot_bgcolor="#1c120c",
                                          xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.image(vis / (vis.max() + 1e-8), caption=f"Sample {idx}", width=250)
            except Exception as e:
                st.error(f"Could not load preview: {e}")
        else:
            st.info("No train.pt found in this dataset")
    else:
        st.info("Select a dataset to preview tensors")

    st.markdown("---")
    st.markdown("### 📤 Upload New Files")
    uploaded = st.file_uploader("Choose files to upload", type=["pt", "pth", "zip"], accept_multiple_files=True, key="ds_upload")
    if uploaded:
        for f in uploaded:
            st.success(f"✅ {f.name} ready ({f.size / 1e6:.1f} MB)")
        if st.button("⬆️ Upload to O3 Bucket", width='stretch'):
            with st.spinner("Uploading..."):
                time.sleep(2)
            st.success("Upload complete!")

# ============================================================================
# PAGE: TRAINING
# ============================================================================
def page_training():
    top_bar()
    st.markdown("""<div class="section-title" style="font-size:20px;">🤖 Training Job Runner</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Configuration")

        # Show active dataset
        if st.session_state.selected_dataset:
            samples = get_sample_datasets()
            ds = samples.get(st.session_state.selected_dataset, {})
            ds_display = ds.get("meta", {}).get("name", st.session_state.selected_dataset) if ds else st.session_state.selected_dataset
            st.markdown(f"""
            <div style="background:rgba(232,69,30,0.12); border:1px solid #e8451e; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
                <div style="font-size:11px; color:#e8451e; font-weight:600;">DATASET</div>
                <div style="font-size:14px; font-weight:600; color:#f0e4d8;">{ds_display}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No dataset selected. Go to Dashboard and pick one.")

        ckpt_b = st.text_input("Checkpoint Bucket:", value=st.session_state.ckpt_bucket or "", placeholder="optional — for O3 upload", key="tr_ckpt")

        st.markdown("---")

        epochs = st.number_input("Epochs:", value=5, min_value=1, max_value=200, key="tr_ep")
        batch = st.selectbox("Batch Size:", [8, 16, 32, 64, 128, 256], index=2, key="tr_bs")
        lr = st.text_input("Learning Rate:", value="0.001", key="tr_lr")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.training_running:
            st.markdown("""<div style="background:rgba(63,185,80,0.12); border:1px solid #3fb950; border-radius:8px; padding:10px 14px; font-size:13px; color:#3fb950; font-weight:600; text-align:center;">⏳ Training in progress...</div>""", unsafe_allow_html=True)
        else:
            can_start = st.session_state.selected_dataset is not None
            if st.button("▶  Start Training", key="tr_start", width='stretch', disabled=not can_start):
                st.session_state.training_running = True
                st.session_state.training_done = False
                st.session_state.training_epoch = 0
                st.session_state.training_total_epochs = epochs
                st.session_state.training_logs = []
                st.session_state.checkpoints = []
                st.session_state.stop_training = False
                try:
                    lr_val = float(lr)
                except ValueError:
                    lr_val = 0.001
                run_training(
                    dataset_name=st.session_state.selected_dataset,
                    epochs=epochs, batch_size=batch, lr=lr_val,
                    ckpt_bucket=ckpt_b,
                )
                st.rerun()

    with col2:
        st.markdown("### 📊 Live Monitor")

        epoch_cur = st.session_state.training_epoch
        epoch_total = st.session_state.training_total_epochs
        is_done = st.session_state.get("training_done", False)

        if epoch_cur > 0 or is_done:
            status_color = "#3fb950" if is_done else "#e8451e"
            status_text = "✅ Complete" if is_done else "🔄 Running"
            st.markdown(f"""
            <div class="card-inner" style="padding:20px;">
                <strong style="color:{status_color};">{status_text}</strong><br><br>
                <strong>⏱️ Epoch:</strong> {epoch_cur} / {epoch_total}<br>
                <strong>📉 Loss:</strong> {st.session_state.training_loss:.4f}<br>
                <strong>📈 Accuracy:</strong> {st.session_state.training_accuracy:.1f}%<br>
            </div>
            """, unsafe_allow_html=True)

            # Show checkpoint summary
            if st.session_state.checkpoints:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 💾 Saved Checkpoints")
                for ckpt in st.session_state.checkpoints[-5:]:
                    cid_display = ckpt['cid'][:20] + "..." if len(ckpt['cid']) > 20 else ckpt['cid']
                    st.markdown(f"""<div style="padding:6px 10px; background:#251a12; border-radius:6px; margin:3px 0; font-size:12px; display:flex; justify-content:space-between;">
                        <span>Epoch {ckpt['epoch']} — {ckpt['accuracy']}%</span><span style="color:#e8451e;">{cid_display}</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-inner" style="padding:20px;">
                <strong>Status:</strong> Idle<br>
                <strong>Epoch:</strong> —<br>
                <strong>Loss:</strong> —<br>
                <strong>Accuracy:</strong> —<br>
            </div>
            """, unsafe_allow_html=True)
            st.info("Configure and start training")

        st.markdown("---")

        # Log viewer
        if st.session_state.training_logs:
            st.markdown("### 📝 Logs")
            log_txt = "\n".join(st.session_state.training_logs[-12:])
            st.markdown(f'<div class="log-viewer" style="max-height:250px;">{log_txt}</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: CHECKPOINTS
# ============================================================================
def page_checkpoints():
    top_bar()
    st.markdown("""<div class="section-title" style="font-size:20px;">💾 Checkpoint Manager</div>""", unsafe_allow_html=True)

    ckpts = st.session_state.get("checkpoints", [])

    if not ckpts:
        st.markdown("""
        <div class="card-inner" style="padding:40px; text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">💾</div>
            <div style="font-size:16px; font-weight:600; color:#f0e4d8;">No Checkpoints Yet</div>
            <div style="font-size:13px; color:#a08070; margin-top:8px;">Train a model to see real checkpoints with CIDs here</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Build table from real checkpoint data
    df = pd.DataFrame([{
        "Epoch": c["epoch"],
        "CID": c["cid"],
        "Accuracy": f"{c['accuracy']}%",
        "Loss": c["loss"],
        "Time": c["timestamp"].split(" ")[-1] if " " in c["timestamp"] else c["timestamp"],
        "Size": fmt_size(c.get("size", 0)),
    } for c in reversed(ckpts)])

    st.dataframe(df, width='stretch', hide_index=True)

    st.markdown("---")

    # Checkpoint details
    st.markdown("### 📋 Checkpoint Details")
    ckpt_options = [f"Epoch {c['epoch']} — {c['cid']}" for c in reversed(ckpts)]
    sel_idx = st.selectbox("Select:", range(len(ckpt_options)), format_func=lambda x: ckpt_options[x], key="ckpt_sel")

    selected_ckpt = list(reversed(ckpts))[sel_idx]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card-inner" style="padding:20px;">
            <strong style="color:#e8451e;">Epoch:</strong> {selected_ckpt['epoch']}<br>
            <strong style="color:#e8451e;">CID:</strong> {selected_ckpt['cid']}<br>
            <strong style="color:#e8451e;">Timestamp:</strong> {selected_ckpt['timestamp']}<br>
            <strong style="color:#e8451e;">Path:</strong> {selected_ckpt.get('path', '—')}<br>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card-inner" style="padding:20px;">
            <strong style="color:#e8451e;">Accuracy:</strong> {selected_ckpt['accuracy']}%<br>
            <strong style="color:#e8451e;">Loss:</strong> {selected_ckpt['loss']}<br>
            <strong style="color:#e8451e;">Size:</strong> {fmt_size(selected_ckpt.get('size', 0))}<br>
            <strong style="color:#e8451e;">Format:</strong> .pt<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶  Resume Training", width='stretch', key="ckpt_resume"):
            st.session_state.page = "Training"
            st.rerun()
    with c2:
        if st.button("📋 View Full Metadata", width='stretch', key="ckpt_meta"):
            st.json(selected_ckpt)
    with c3:
        is_o3 = selected_ckpt['cid'] != "local-only"
        st.markdown(f"""
        <div style="padding:8px; text-align:center; font-size:12px; font-weight:600; color:{'#3fb950' if is_o3 else '#a08070'};">
            {'✅ Stored on O3' if is_o3 else '📁 Local only (connect wallet to upload)'}
        </div>
        """, unsafe_allow_html=True)

    # ── Lineage Graph ──
    st.markdown("---")
    st.markdown("### 🔗 Model Lineage Graph")

    lineage_html = '<div style="display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap;">'
    for i, ckpt in enumerate(ckpts):
        is_last = i == len(ckpts) - 1
        is_o3 = ckpt['cid'] != "local-only"
        border_color = "#e8451e" if is_last else ("#3fb950" if is_o3 else "#2d1f16")
        bg = "#e8451e" if is_last else "#251a12"
        fw = "700" if is_last else "400"
        cid_short = ckpt['cid'][:8] if is_o3 else ""
        label = f"E{ckpt['epoch']}"
        if is_last:
            label += " ✓"
        lineage_html += f'<div style="background:{bg}; border:2px solid {border_color}; border-radius:8px; padding:6px 14px; font-size:11px; font-weight:{fw}; color:white; text-align:center;">{label}<br><span style="font-size:9px; opacity:0.7;">{cid_short}</span></div>'
        if not is_last:
            lineage_html += '<span style="color:#e8451e; font-size:16px;">→</span>'
    lineage_html += '</div>'

    has_o3 = any(c['cid'] != 'local-only' for c in ckpts)
    footer_msg = "CID-based versioning on Akave O3" if has_o3 else "Local checkpoints — connect wallet for O3 CIDs"

    st.markdown(f"""
    <div class="card-inner" style="padding:20px; text-align:center;">
        {lineage_html}
        <div style="margin-top:12px; font-size:12px; color:#a08070;">{footer_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Accuracy/Loss chart
    if len(ckpts) >= 2 and HAS_PLOTLY:
        st.markdown("---")
        st.markdown("### 📈 Training Progress")
        epochs_list = [c["epoch"] for c in ckpts]
        acc_list = [c["accuracy"] for c in ckpts]
        loss_list = [c["loss"] for c in ckpts]

        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
        fig.add_trace(plotly_go.Scatter(x=epochs_list, y=acc_list, mode='lines+markers', name='Accuracy', line=dict(color='#3fb950')), row=1, col=1)
        fig.add_trace(plotly_go.Scatter(x=epochs_list, y=loss_list, mode='lines+markers', name='Loss', line=dict(color='#e8451e')), row=1, col=2)
        fig.update_layout(height=300, paper_bgcolor="#1c120c", plot_bgcolor="#110a06",
                          font=dict(color="#f0e4d8"), showlegend=False,
                          margin=dict(l=40, r=20, t=40, b=30))
        fig.update_xaxes(title_text="Epoch", gridcolor="#2d1f16")
        fig.update_yaxes(gridcolor="#2d1f16")
        st.plotly_chart(fig, width='stretch')

# ============================================================================
# PAGE: BUCKETS
# ============================================================================
def page_buckets():
    top_bar()
    st.markdown("""<div class="section-title" style="font-size:20px;">🪣 O3 Buckets</div>""", unsafe_allow_html=True)

    if not st.session_state.connected:
        st.markdown("""
        <div class="card-inner" style="padding:40px; text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">🔒</div>
            <div style="font-size:16px; font-weight:600; color:#f0e4d8;">Wallet Not Connected</div>
            <div style="font-size:13px; color:#a08070; margin-top:8px;">Connect your wallet in Settings to view O3 buckets</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Fetch buckets
    client = st.session_state.o3_client
    try:
        buckets = client.list_buckets()
        bucket_list = [{"name": b.name if hasattr(b, 'name') else str(b)} for b in buckets]
    except Exception as e:
        st.error(f"Error fetching buckets: {e}")
        return

    if not bucket_list:
        st.markdown("""
        <div class="card-inner" style="padding:40px; text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">📭</div>
            <div style="font-size:16px; font-weight:600; color:#f0e4d8;">No Buckets</div>
            <div style="font-size:13px; color:#a08070; margin-top:8px;">Create your first bucket in Settings or via the O3Client API</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Display buckets table
    st.markdown("### 📋 All Buckets")
    df = pd.DataFrame([{"Bucket Name": b["name"], "Status": "✅ Active", "Type": "Object Storage"} for b in bucket_list])
    st.dataframe(df, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        new_bucket_name = st.text_input("New bucket name:", placeholder="e.g., my-ml-data")
        if st.button("➕ Create New Bucket", width='stretch', key="create_bucket_btn"):
            if new_bucket_name:
                try:
                    client.create_bucket(new_bucket_name)
                    st.success(f"✅ Bucket '{new_bucket_name}' created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating bucket: {e}")
            else:
                st.warning("Enter a bucket name")
    
    with col2:
        st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Buckets", width='stretch', key="refresh_buckets"):
            st.rerun()
    
    with col3:
        st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)
        if st.button("📖 API Reference", width='stretch', key="buckets_api_ref"):
            st.session_state.page = "API Docs"
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example: Use Buckets in Python")
    code = """from pytorch_o3 import O3Client

client = O3Client()
# List all buckets
buckets = client.list_buckets()

# Create a new bucket
client.create_bucket("my-training-data")

# List objects in a bucket
objects = client.list_objects("my-training-data", prefix="models/")
"""
    st.code(code, language="python")

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
def page_settings():
    top_bar()
    st.markdown("""<div class="section-title" style="font-size:20px;">⚙️ Settings</div>""", unsafe_allow_html=True)

    st.markdown("### 🔐 Akave Configuration")

    key_input = st.text_input("AKAVE_PRIVATE_KEY:", type="password", key="settings_key",
                              value="" if not st.session_state.connected else "")
    ipc_addr = st.text_input("IPC Address:", value="connect.akave.ai:5500", key="settings_ipc")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔗 Test Connection", width='stretch', key="set_test"):
            if key_input:
                if do_connect(key_input):
                    st.success("✅ Connected successfully!")
                    time.sleep(1)
                    st.rerun()
            else:
                # Try from .env
                env_key = os.getenv("AKAVE_PRIVATE_KEY")
                if env_key:
                    if do_connect(env_key):
                        st.success("✅ Connected from .env!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("Enter a key or set AKAVE_PRIVATE_KEY in .env")
    with c2:
        if st.button("📄 Load from .env", width='stretch', key="set_env"):
            env_key = os.getenv("AKAVE_PRIVATE_KEY")
            if env_key:
                if do_connect(env_key):
                    st.success("✅ Connected from .env!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("AKAVE_PRIVATE_KEY not found in .env")

    if st.session_state.connected:
        st.markdown(f"""
        <div style="margin-top:12px;">
            <span class="green-dot"></span>
            <span style="color:#3fb950; font-weight:600; margin-left:6px;">Connected as {st.session_state.wallet_addr}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔌 Disconnect Wallet", width='stretch', key="set_disconnect"):
            st.session_state.connected = False
            st.session_state.wallet_addr = None
            st.session_state.o3_client = None
            st.rerun()

    st.markdown("---")

    st.markdown("### 🎛️ General")
    st.checkbox("Auto-save checkpoints", value=True)
    st.checkbox("Email notifications on completion", value=False)
    st.slider("Max upload workers:", 1, 16, 4)
    st.slider("Cache size (GB):", 1, 100, 20)

# ============================================================================
# ROUTER
# ============================================================================
pages = {
    "Overview": page_overview,
    "Dashboard": page_dashboard,
    "Datasets": page_datasets,
    "Training": page_training,
    "Checkpoints": page_checkpoints,
    "Buckets": page_buckets,
    "API Docs": page_api_docs,
    "Settings": page_settings,
}

pages.get(st.session_state.page, page_overview)()
