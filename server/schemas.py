from pydantic import BaseModel, Field
from typing import List, Optional, Union

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: str = "float" 

class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "pure-onnx"

class ModelDeletionResponse(BaseModel):
    id: str
    object: str = "model"
    deleted: bool

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
