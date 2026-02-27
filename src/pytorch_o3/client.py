import os
import io
import logging

from akavesdk import SDK, SDKConfig, SDKError
from tenacity import retry, wait_exponential, stop_after_attempt

from .exceptions import O3AuthError

logger = logging.getLogger(__name__)

class O3Client:
    def __init__(self, private_key=None, ipc_address="connect.akave.ai:5500"):
        self.private_key = private_key or os.getenv('AKAVE_PRIVATE_KEY')
        self._ipc_address = ipc_address
        if not self.private_key:
            raise O3AuthError("AKAVE_PRIVATE_KEY is missing.")
            
        try:
            config = SDKConfig(
                address=ipc_address,
                max_concurrency=10,
                block_part_size=1_000_000,
                private_key=self.private_key,
                use_connection_pool=True,
                connection_timeout=30
            )
            self.sdk = SDK(config)
            self.ipc = self.sdk.ipc()
        except SDKError as e:
            raise O3AuthError(f"SDK Error: {e}")
        except Exception as e:
            raise O3AuthError(f"Init Error: {e}")
        
    def create_bucket(self,name):
        return self.ipc.create_bucket(None, name)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5)) 
    def list_buckets(self):
        try:
            return self.ipc.list_buckets(None, offset=0, limit=0)
        except Exception as e:
            raise e
    
    def list_objects(self, bucket_name: str, prefix: str = "", limit: int = 1000):
        if hasattr(self.ipc, 'list_files'):
            files = self.ipc.list_files(None, bucket_name)
            
            if prefix:
                files = [f for f in files if hasattr(f, 'name') and f.name.startswith(prefix)]
                if len(files) < len([f for f in self.ipc.list_files(None, bucket_name) if hasattr(f, 'name')]):
                    logger.warning(f"Some objects may have been filtered out due to missing 'name' attribute")
            
            if limit and limit > 0:
                files = files[:limit]
            
            return files
        else:
            raise NotImplementedError("akavesdk IPC interface needs list_files method")
    
    def get_object_info(self, bucket_name: str, key: str):
        if hasattr(self.ipc, 'file_info'):
            info = self.ipc.file_info(None, bucket_name, key)
            if info is None:
                raise RuntimeError(f"File {key} not found in bucket {bucket_name}")
            return info
        else:
            raise NotImplementedError("akavesdk IPC interface needs file_info method")
    
    def download_object_range(self, bucket_name: str, key: str, start: int, end: int) -> bytes:
        if hasattr(self.ipc, 'create_range_file_download') and hasattr(self.ipc, 'download'):
            file_download = self.ipc.create_range_file_download(None, bucket_name, key, start, end)
            buffer = io.BytesIO()
            self.ipc.download(None, file_download, buffer)
            return buffer.getvalue()
        elif hasattr(self.ipc, 'download'):
            file_download = self.ipc.create_file_download(None, bucket_name, key)
            buffer = io.BytesIO()
            self.ipc.download(None, file_download, buffer)
            full_data = buffer.getvalue()
            return full_data[start:end]
        else:
            raise NotImplementedError("akavesdk IPC interface needs create_range_file_download and download methods")

    def close(self):
        self.sdk.close()
