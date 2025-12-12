import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from embeddings.embedder import ONNXEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

class PredictionRequest(BaseModel):
    texts: list[str]
    batch_size: int = 32

embedder: ONNXEmbedder | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    
    # Load config from environment variables
    model_path = os.environ.get("ZEPHYR_MODEL_PATH")
    tokenizer_path = os.environ.get("ZEPHYR_TOKENIZER_PATH")
    max_length = int(os.environ.get("ZEPHYR_MAX_LENGTH", "512"))
    device = os.environ.get("ZEPHYR_DEVICE", "cpu")
    provider = os.environ.get("ZEPHYR_PROVIDER")

    if not model_path or not tokenizer_path:
        logger.error("Missing ZEPHYR_MODEL_PATH or ZEPHYR_TOKENIZER_PATH env vars")
        sys.exit(1)

    logger.info(f"Loading model from {model_path} on {device}...")
    try:
        embedder = ONNXEmbedder(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            device=device,
            provider=provider
        )
        logger.info("Model loaded successfully.")
        
        # Warmup: Run a dummy inference to trigger lazy compilation (MIGraphX/ROCm)
        if device == "gpu":
            logger.info("Running warmup inference to compile GPU kernels...")
            try:
                # Use a long sequence to trigger max-shape compilation
                # MIGraphX often recompiles for larger shapes if not seen before
                warmup_text = "warmup " * (max_length // 2)
                embedder.predict([warmup_text])
                logger.info("Warmup complete.")
            except Exception as e:
                logger.warning(f"Warmup failed (non-fatal): {e}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't exit here immediately to allow logs to be flushed, but health check will fail
        # Actually better to raise or exit
        sys.exit(1)
        
    yield
    
    # improved cleanup if needed
    embedder = None

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not embedder:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        if len(request.texts) > request.batch_size:
             embeddings, tokens = embedder.predict_batched(request.texts, batch_size=request.batch_size)
        else:
             embeddings, tokens = embedder.predict(request.texts)
        return {"embeddings": embeddings, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    if embedder:
        return {"status": "ok", "device": "ready"}
    return {"status": "loading"}
