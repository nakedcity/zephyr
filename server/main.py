from contextlib import asynccontextmanager
from pathlib import Path
import sys
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

from config.loader import load_config
from embeddings.model_cache import ModelCache
from server.logging_config import setup_logging
from server.middleware import TraceIDMiddleware
from server.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingObject,
    ModelList,
    ModelCard,
    ModelDeletionResponse,
    Usage,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
auth_scheme = HTTPBearer(auto_error=False)
config = load_config(CONFIG_PATH)

# Setup logging with trace ID support
logger = setup_logging()


def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if config['authorization']['enabled'] is True:
        """
        Validate Authorization: Bearer <token> using token_env_var from authorization conf.
        Mimics OpenAI's bearer token requirement.
        """
        api_token = os.getenv(config['authorization']['token_env_var'])
        if not api_token:
            raise HTTPException(status_code=500, detail="Server API key not configured")

        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

        if credentials.credentials != api_token:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return True
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cache = ModelCache(config)
    
    # Pre-download models in preload list
    logger.info("Preloading models...")
    for model_id in config.preload:
        try:
            logger.info(f"Loading {model_id}...")
            cache.get_model(model_id)
            logger.info(f"Loaded {model_id}")
        except Exception as e:
            logger.error(f"Failed to preload {model_id}: {e}")
    
    app.state.cache = cache
    app.state.config = config
    
    yield
    
    # Shutdown
    cache.clear_all()

app = FastAPI(lifespan=lifespan)

# Add trace ID middleware
app.add_middleware(TraceIDMiddleware)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, _: bool = Security(verify_bearer_token)):
    import time
    
    cache: ModelCache = app.state.cache
    
    try:
        embedder = cache.get_model(request.model)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]

    # Log batch size
    num_inputs = len(inputs)
    logger.info(f"EMBEDDING REQUEST: Received {num_inputs} texts for model '{request.model}'")
    
    # Get batch_size from model config, default to 64 if not specified
    model_config = app.state.config.models.get(request.model)
    batch_size = getattr(model_config, 'batch_size', 64)
    
    try:
        start_time = time.time()
        embeddings = embedder.predict_batched(inputs, batch_size=batch_size)
        elapsed = time.time() - start_time
        logger.info(f"EMBEDDING COMPLETE: Processed {num_inputs} texts in {elapsed:.2f}s ({num_inputs/elapsed:.1f} texts/sec) [batch_size={batch_size}]")
    except Exception as e:
        logger.error(f"EMBEDDING ERROR: Failed to process {num_inputs} texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Construct response
    data = []
    for i, emb in enumerate(embeddings):
        data.append(EmbeddingObject(embedding=emb, index=i))

    # Usage stats (approximate)
    usage = Usage(prompt_tokens=0, total_tokens=0)

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=usage
    )

@app.get("/v1/models", response_model=ModelList)
async def list_models(_: bool = Security(verify_bearer_token)):
    config = app.state.config
    cache: ModelCache = app.state.cache
    models = []
    for model_id, model_conf in config.models.items():
        owner = getattr(model_conf, "owner", None) or getattr(model_conf, "owned_by", None) or "unknown"

        created = 0
        if hasattr(cache, "get_created_timestamp"):
            try:
                created = int(cache.get_created_timestamp(model_id))
            except Exception:
                created = 0

        models.append(ModelCard(id=model_id, owned_by=owner, created=created))
    return ModelList(data=models)

@app.get("/v1/models/{model_id}", response_model=ModelCard)
async def retrieve_model(model_id: str, _: bool = Security(verify_bearer_token)):
    config = app.state.config
    cache: ModelCache = app.state.cache

    if model_id not in config.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Attempt to load the model (or get from cache if already loaded)
    try:
        cache.get_model(model_id)
    except ValueError:
        # Should not happen because we already verified config
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Prepare response metadata
    model_conf = config.models[model_id]
    owner = getattr(model_conf, "owner", None) or getattr(model_conf, "owned_by", None) or "unknown"
    created = 0
    if hasattr(cache, "get_created_timestamp"):
        try:
            created = int(cache.get_created_timestamp(model_id))
        except Exception:
            created = 0

    return ModelCard(id=model_id, owned_by=owner, created=created)

@app.delete("/v1/models/{model_id}", response_model=ModelDeletionResponse)
async def delete_model(model_id: str, _: bool = Security(verify_bearer_token)):
    config = app.state.config
    cache: ModelCache = app.state.cache

    if model_id not in config.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    deleted = False
    if hasattr(cache, "unload_model"):
        try:
            deleted = bool(cache.unload_model(model_id))
        except Exception:
            deleted = False

    return ModelDeletionResponse(id=model_id, deleted=deleted)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    config = load_config(CONFIG_PATH)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
