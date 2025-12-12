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
    # batch_size is handled server-side based on config

embedder: ONNXEmbedder | None = None
processing_batch_size: int = 32

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, processing_batch_size
    
    # Load config from environment variables
    model_path = os.environ.get("ZEPHYR_MODEL_PATH")
    tokenizer_path = os.environ.get("ZEPHYR_TOKENIZER_PATH")
    max_length = int(os.environ.get("ZEPHYR_MAX_LENGTH", "512"))
    device = os.environ.get("ZEPHYR_DEVICE", "cpu")
    provider = os.environ.get("ZEPHYR_PROVIDER")
    
    # Batch size config (resolved by process_manager to be static_batch_size if applicable)
    processing_batch_size = int(os.environ.get("ZEPHYR_BATCH_SIZE", "32"))
    
    # Static batch size for avoiding recompilation (e.g. MIGraphX)
    # We use the generic batch_size for this purpose if we are on MIGraphX
    static_batch_size = processing_batch_size if device == "gpu" and provider == "migraphx" else None

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
            provider=provider,
            static_batch_size=static_batch_size
        )
        logger.info(f"Model loaded successfully. Processing batch size: {processing_batch_size}")
        
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
        # Force immediate exit to prevent hanging processes
        # sys.exit(1) raises SystemExit which can be caught by Uvicorn/Starlette
        # os._exit(1) terminates the process immediately
        os._exit(1)
        
    yield
    
    # improved cleanup if needed
    embedder = None

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not embedder:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Always use predict_batched to ensure consistent batching/padding behavior
        embeddings, tokens = embedder.predict_batched(request.texts, batch_size=processing_batch_size)
        return {"embeddings": embeddings, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    if embedder:
        return {"status": "ok", "device": "ready"}
    return {"status": "loading"}
