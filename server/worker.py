import os
import sys
import argparse
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from embeddings.embedder import ONNXEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

class ModelConfig(BaseModel):
    model_path: str
    tokenizer_path: str
    max_length: int
    device: str
    provider: str | None = None

class PredictionRequest(BaseModel):
    texts: list[str]
    batch_size: int = 32

app = FastAPI()
embedder: ONNXEmbedder | None = None

def load_embedder(args):
    global embedder
    logger.info(f"Loading model from {args.model_path} on {args.device}...")
    try:
        embedder = ONNXEmbedder(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            max_length=args.max_length,
            device=args.device,
            provider=args.provider
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not embedder:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        if len(request.texts) > request.batch_size:
             embeddings = embedder.predict_batched(request.texts, batch_size=request.batch_size)
        else:
             embeddings = embedder.predict(request.texts)
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    if embedder:
        return {"status": "ok", "device": "ready"}
    return {"status": "loading"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, required=True) # cpu or gpu
    parser.add_argument("--provider", type=str, default=None) # cuda or rocm
    
    args = parser.parse_args()
    
    load_embedder(args)
    
    uvicorn.run(app, host="127.0.0.1", port=args.port)
