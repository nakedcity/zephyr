import os
import time
from collections import OrderedDict
from huggingface_hub import hf_hub_download
from embeddings.embedder import ONNXEmbedder
from embeddings.quantizer import quantize_model

class ModelCache:
    def __init__(self, config):
        self.config = config
        self.cache_dir = config.cache.directory
        self.max_loaded = config.cache.max_loaded_models
        # LRU cache: key=model_id, value=ONNXEmbedder
        self.loaded_models = OrderedDict()
        # Track metadata like created timestamp per loaded model
        self.loaded_metadata = {}

    def get_model(self, model_id: str) -> ONNXEmbedder:
        if model_id not in self.config.models:
            raise ValueError(f"Model {model_id} not configured")

        # Check if loaded
        if model_id in self.loaded_models:
            self.loaded_models.move_to_end(model_id)
            return self.loaded_models[model_id]

        # Evict if full
        if len(self.loaded_models) >= self.max_loaded:
            evicted_id, _ = self.loaded_models.popitem(last=False)
            self.loaded_metadata.pop(evicted_id, None)

        # Load model
        model_conf = self.config.models[model_id]
        repo_id = model_conf.repo
        
        # Download files
        # Try to find model.onnx in root or onnx/ subfolder
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx", cache_dir=self.cache_dir)
        except Exception:
            # Try onnx/ subfolder (common in Xenova repos)
            print(f"model.onnx not found in root, trying onnx/model.onnx...")
            model_path = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx", cache_dir=self.cache_dir)

        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir=self.cache_dir)
        
        # Also download config.json just in case
        try:
            hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=self.cache_dir)
        except Exception:
            pass

        # Handle Quantization
        quantize = getattr(model_conf, 'quantize', False)
        if quantize:
            quantized_filename = "model_quantized.onnx"
            quantized_path = os.path.join(os.path.dirname(model_path), quantized_filename)
            
            if not os.path.exists(quantized_path):
                print(f"Quantizing model {model_id}...")
                quantize_model(model_path, quantized_path)
            
            model_path = quantized_path

        # Get device config, default to cpu
        device = getattr(model_conf, 'device', 'cpu')

        embedder = ONNXEmbedder(model_path, tokenizer_path, max_length=model_conf.max_tokens, device=device)
        self.loaded_models[model_id] = embedder
        self.loaded_metadata[model_id] = {"created": int(time.time())}
        return embedder

    def get_created_timestamp(self, model_id: str) -> int:
        """Return the UNIX timestamp (seconds) when the model was loaded, or 0 if not loaded."""
        return self.loaded_metadata.get(model_id, {}).get("created", 0)

    def clear_all(self):
        self.loaded_models.clear()
        self.loaded_metadata.clear()
