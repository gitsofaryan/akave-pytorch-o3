import os

from akavesdk import SDK, SDKConfig, SDKError
from tenacity import retry, wait_exponential, stop_after_attempt

from .exceptions import O3AuthError

class O3Client:
    def __init__(self, private_key=None, ipc_address="connect.akave.ai:5500"):
        self.private_key = private_key or os.getenv('AKAVE_PRIVATE_KEY')
        if not self.private_key:
            raise O3AuthError("AKAVE_PRIVATE_KEY is missing.")
            
        try:
            config = SDKConfig(
                address=ipc_address,
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

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def list_buckets(self):
        try:
            return self.ipc.list_buckets(None, offset=0, limit=0)
        except Exception as e:
            raise e

    def close(self):
        self.sdk.close()
