import requests
import json

class RemoteEmbedder:
    def __init__(self, port: int, model_id: str):
        self.base_url = f"http://127.0.0.1:{port}"
        self.model_id = model_id
        
        # Verify connection
        try:
             resp = requests.get(f"{self.base_url}/health", timeout=5)
             resp.raise_for_status()
             if resp.json().get("status") != "ok":
                  raise RuntimeError("Worker reported unhealthy status")
        except Exception as e:
             raise RuntimeError(f"Failed to connect to worker at {self.base_url}: {e}")

    def predict(self, texts: list[str]) -> list[list[float]]:
        # This matches the signature of ONNXEmbedder.predict
        # But we'll implement batched prediction on the server side or client side?
        # The worker endpoint handles batching if we send it all.
        # But let's reuse the simple batching logic here or trust the worker.
        # References server/worker.py: /predict accepts texts and batch_size
        
        response = requests.post(
            f"{self.base_url}/predict",
            json={"texts": texts, "batch_size": 32}, # Use default batch size or pass it in?
             timeout=60 # Long timeout for processing
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def predict_batched(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
         # We can just delegate to the worker
        response = requests.post(
            f"{self.base_url}/predict",
            json={"texts": texts, "batch_size": batch_size},
             timeout=300
        )
        response.raise_for_status()
        return response.json()["embeddings"]
