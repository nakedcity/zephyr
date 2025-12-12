import httpx
import logging

logger = logging.getLogger(__name__)

class RemoteEmbedder:
    def __init__(self, port: int, model_id: str):
        self.base_url = f"http://127.0.0.1:{port}"
        self.model_id = model_id
        
        # Verify connection synchronously
        try:
             with httpx.Client(timeout=5) as client:
                 resp = client.get(f"{self.base_url}/health")
                 resp.raise_for_status()
                 if resp.json().get("status") != "ok":
                      raise RuntimeError("Worker reported unhealthy status")
        except Exception as e:
             raise RuntimeError(f"Failed to connect to worker at {self.base_url}: {e}")

        # Initialize async client for predictions
        # Note: Ideally this should be closed, but it lives for the app lifetime
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300)

    async def predict(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.post(
                "/predict",
                json={"texts": texts, "batch_size": 32}
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPError as e:
            logger.error(f"Prediction failed: {e}")
            raise

    async def predict_batched(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        try:
            response = await self.client.post(
                "/predict",
                json={"texts": texts, "batch_size": batch_size}
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPError as e:
            logger.error(f"Prediction failed: {e}")
            raise
