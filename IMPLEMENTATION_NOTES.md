# O3Dataset Implementation Notes

O3Dataset provides a PyTorch Dataset for streaming data from Akave O3 storage with range-based downloads and LRU caching.

## API Integration

The implementation uses the following akavesdk methods:

- `ipc.list_files(ctx, bucket_name)` - list objects in bucket
- `ipc.file_info(ctx, bucket_name, file_name)` - get object metadata (returns IPCFileMeta with .size)
- `ipc.create_range_file_download(ctx, bucket_name, file_name, start, end)` - create range download
- `ipc.download(ctx, file_download, writer)` - download to BytesIO buffer

## Usage

```python
from pytorch_o3 import O3Client, O3Dataset
from torch.utils.data import DataLoader

client = O3Client()
dataset = O3Dataset(
    client=client,
    bucket_name="my-bucket",
    object_keys=["data/file1.pt", "data/file2.pt"],
    chunk_size=1024 * 1024,
    cache_size=100,
    cache_dir="./cache"
)

dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
for batch in dataloader:
    # training code
    pass
```

## Architecture

- LRU cache: Thread-safe, evicts least recently used chunks
- Chunking: Objects divided into fixed-size chunks, downloaded on-demand
- Caching: Memory cache + optional disk cache to reduce network calls
- DataLoader: Fully compatible with PyTorch DataLoader, supports multiple workers
