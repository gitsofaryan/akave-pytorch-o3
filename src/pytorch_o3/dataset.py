"""PyTorch dataset for streaming data from Akave O3 storage."""

import os
import hashlib
import logging
import threading
import tempfile
from collections import OrderedDict
from typing import Optional, List, Tuple, Any, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, get_worker_info

from .client import O3Client
from .exceptions import O3AuthError

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache."""
    
    def __init__(self, max_size: int):
        if max_size <= 0:
            max_size = 0
        self.max_size = max_size
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[bytes]:
        if self.max_size == 0:
            return None
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: bytes) -> None:
        if self.max_size == 0:
            return
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def __getstate__(self):
        return {'max_size': self.max_size}
    
    def __setstate__(self, state):
        self.__init__(state['max_size'])
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)


class O3Dataset(Dataset):
    """PyTorch dataset for streaming data from Akave O3 storage."""
    
    def __init__(
        self,
        client: O3Client,
        bucket_name: str,
        object_keys: List[str],
        chunk_size: int = 1024 * 1024,
        cache_size: int = 100,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ):
        if not object_keys:
            raise ValueError("object_keys cannot be empty")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        
        self._client = client
        self._private_key = client.private_key
        self._ipc_address = getattr(client, '_ipc_address', "connect.akave.ai:5500")
        self.bucket_name = bucket_name
        self.object_keys = object_keys
        self.chunk_size = chunk_size
        self.transform = transform
        self.target_transform = target_transform
        
        self.cache = LRUCache(max_size=cache_size)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self._object_metadata = {}
        self._compute_metadata()
    
    @property
    def client(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            if not hasattr(self, '_worker_clients'):
                self._worker_clients = {}
            if worker_id not in self._worker_clients:
                self._worker_clients[worker_id] = O3Client(
                    private_key=self._private_key,
                    ipc_address=self._ipc_address
                )
            return self._worker_clients[worker_id]
        return self._client
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_client'] = None
        state['_worker_clients'] = {}
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = O3Client(
            private_key=self._private_key,
            ipc_address=self._ipc_address
        )
    
    def _compute_metadata(self) -> None:
        for key in self.object_keys:
            try:
                size = self._get_object_size(key)
                num_chunks = (size + self.chunk_size - 1) // self.chunk_size
                self._object_metadata[key] = {
                    'size': size,
                    'num_chunks': num_chunks,
                    'chunk_size': self.chunk_size
                }
            except Exception as e:
                raise RuntimeError(f"Failed to get metadata for object {key}: {e}") from e
    
    def _get_object_size(self, key: str) -> int:
        info = self.client.get_object_info(self.bucket_name, key)
        
        if hasattr(info, 'size') and info.size is not None:
            return int(info.size)
        
        for attr_name in ['Size', 'file_size', 'length', 'fileLength']:
            if hasattr(info, attr_name):
                size = getattr(info, attr_name)
                if size is not None:
                    return int(size)
        
        if isinstance(info, dict):
            size = info.get('size')
            if size is None:
                size = info.get('Size')
            if size is not None:
                return int(size)
        
        raise ValueError(f"Could not extract size from object info for {key}")
    
    def _get_cache_key(self, key: str, chunk_idx: int) -> str:
        return f"{key}:{chunk_idx}"
    
    def _get_disk_cache_path(self, cache_key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        return Path(self.cache_dir) / f"{cache_hash}.chunk"
    
    def _download_range(self, key: str, start: int, end: int) -> bytes:
        return self.client.download_object_range(self.bucket_name, key, start, end)
    
    def _get_chunk(self, key: str, chunk_idx: int) -> bytes:
        cache_key = self._get_cache_key(key, chunk_idx)
        
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        if self.cache_dir:
            disk_path = self._get_disk_cache_path(cache_key)
            if disk_path.exists():
                with open(disk_path, 'rb') as f:
                    disk_data = f.read()
                self.cache.put(cache_key, disk_data)
                return disk_data
        
        metadata = self._object_metadata[key]
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, metadata['size'])
        
        chunk_data = self._download_range(key, start, end)
        
        self.cache.put(cache_key, chunk_data)
        if self.cache_dir:
            disk_path = self._get_disk_cache_path(cache_key)
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as tmp:
                tmp.write(chunk_data)
                tmp_path = tmp.name
            os.replace(tmp_path, disk_path)
        
        return chunk_data
    
    
    def __len__(self) -> int:
        return len(self.object_keys)
    
    def _load_full_object(self, key: str) -> bytes:
        metadata = self._object_metadata[key]
        num_chunks = metadata['num_chunks']
        
        if num_chunks == 1:
            return self._get_chunk(key, 0)
        
        chunks = []
        for chunk_idx in range(num_chunks):
            chunk_data = self._get_chunk(key, chunk_idx)
            chunks.append(chunk_data)
        
        return b''.join(chunks)
    
    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self.object_keys):
            raise IndexError(f"Index {idx} out of range [0, {len(self.object_keys)})")
        
        key = self.object_keys[idx]
        object_data = self._load_full_object(key)
        
        if self.transform:
            data = self.transform(object_data)
            if self.target_transform:
                return data, self.target_transform(data)
            return data
        return object_data
    
    def get_cache_stats(self) -> dict:
        return {
            'memory_cache_size': self.cache.size(),
            'memory_cache_max': self.cache.max_size,
            'disk_cache_dir': self.cache_dir,
            'disk_cache_files': len(list(Path(self.cache_dir).glob('*.chunk'))) if self.cache_dir else 0
        }
    
    def clear_cache(self) -> None:
        self.cache.clear()
        if self.cache_dir:
            for cache_file in Path(self.cache_dir).glob('*.chunk'):
                try:
                    cache_file.unlink()
                except (FileNotFoundError, PermissionError, OSError):
                    pass
