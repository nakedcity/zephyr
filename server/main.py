from contextlib import asynccontextmanager
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

from config.loader import load_config
from embeddings.model_cache import ModelCache
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


def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """
    Validate Authorization: Bearer <token> using OPENAI_API_KEY (or API_KEY) env var.
    Mimics OpenAI's bearer token requirement.
    """
    expected = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server API key not configured")

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = credentials.credentials
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config(CONFIG_PATH)
    cache = ModelCache(config)
    
    # Pre-download models in preload list
    print("Preloading models...")
    for model_id in config.preload:
        try:
            print(f"Loading {model_id}...")
            cache.get_model(model_id)
            print(f"Loaded {model_id}")
        except Exception as e:
            print(f"Failed to preload {model_id}: {e}")
    
    app.state.cache = cache
    app.state.config = config
    
    yield
    
    # Shutdown
    cache.clear_all()

app = FastAPI(lifespan=lifespan)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, _: bool = Security(verify_bearer_token)):
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

    try:
        embeddings = embedder.predict(inputs)
    except Exception as e:
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
