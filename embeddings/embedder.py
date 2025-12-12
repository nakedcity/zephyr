
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

# Preload CUDA/cuDNN DLLs from Nvidia site packages if available (for onnxruntime-gpu >= 1.21)
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(directory="")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to preload CUDA DLLs: {e}")

class ONNXEmbedder:
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 512, device: str = "cpu", provider: str | None = None, static_batch_size: int | None = None):
        self.static_batch_size = static_batch_size
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Enable truncation and padding
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length)

        if device == "gpu":
            if provider == "migraphx":
                # ROCm 7.1+ uses MIGraphX as the backend. ROCMExecutionProvider is not available in these wheels.
                providers = ['MIGraphXExecutionProvider']
            elif provider == "rocm":
                # Standard ROCm execution provider
                providers = ['ROCMExecutionProvider']
            elif provider == "cuda":
                providers = ['CUDAExecutionProvider']
            else:
                 # Should fail before this, but safe fallback logic for weird values
                 raise ValueError(f"Unsupported gpu_provider: {provider}")
        else:
            providers = ['CPUExecutionProvider']
            
        print(f"Requesting load on {device} (provider={provider}) with providers: {providers}")
        
        # Suppress warnings (like GPU discovery failure)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        
        try:
             self.session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        except Exception as e:
             raise RuntimeError(f"Failed to initialize session with providers {providers}: {e}")

        # Verify actual providers
        active_providers = self.session.get_providers()
        print(f"Model loaded. Active providers: {active_providers}")
        
        if device == "gpu":
             # Double check that we didn't silently fall back if silent fallback is enabled in ORT (it shouldn't be with our list)
             # But if user has only CPU provider, ORT might still load CPU if it can't find others? 
             # Actually, if we pass only ['CUDAExecutionProvider'], ORT should fail if it can't use it?
             # Let's verify.
             if provider == "migraphx" and not any(p in active_providers for p in ["ROCMExecutionProvider", "MIGraphXExecutionProvider"]):
                 raise RuntimeError(
                    f"GPU requested (provider=migraphx) but neither ROCMExecutionProvider nor MIGraphXExecutionProvider active. "
                    f"Active providers: {active_providers}."
                )
             if provider == "rocm" and "ROCMExecutionProvider" not in active_providers:
                 raise RuntimeError(
                    f"GPU requested (provider=rocm) but ROCMExecutionProvider not active. "
                    f"Active providers: {active_providers}."
                )
             if provider == "cuda" and "CUDAExecutionProvider" not in active_providers:
                raise RuntimeError(
                    f"GPU requested (provider=cuda) but CUDAExecutionProvider not active. "
                    f"Active providers: {active_providers}."
                )
        
    def predict(self, texts: list[str]) -> tuple[list[list[float]], int]:
        # Handle static batching for ROCm/MIGraphX to prevent recompilation
        # We always pad the batch to self.static_batch_size if set
        original_len = len(texts)
        if hasattr(self, 'static_batch_size') and self.static_batch_size:
            if len(texts) > self.static_batch_size:
                 # If input exceeds static size, we must split it or error. 
                 # predict_batched handles splitting, so here we assume it fits or warn.
                 # Actually, let's just process it and let recompilation happen if it exceeds,
                 # but for small batches we pad up.
                 pass
            elif len(texts) < self.static_batch_size:
                 # Pad with empty strings
                 texts = texts + [""] * (self.static_batch_size - len(texts))
        
        # Tokenize
        encoded = self.tokenizer.encode_batch(texts)
        
        # Calculate total tokens (excluding padding) for the original texts only
        # We need to be careful not to count tokens from dummy inputs
        total_tokens = 0
        for i in range(original_len):
            total_tokens += sum(encoded[i].attention_mask)
        
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

        # Run inference
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        
        # Remove token_type_ids if not in model inputs
        model_inputs = [x.name for x in self.session.get_inputs()]
        if 'token_type_ids' not in model_inputs:
            del inputs['token_type_ids']
            
        outputs = self.session.run(None, inputs)
        
        # Usually the last_hidden_state is the first output
        last_hidden_state = outputs[0]
        
        # Mean Pooling
        embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        
        # Normalize
        embeddings = self.normalize(embeddings)
        
        # Slice back to original length
        return embeddings[:original_len].tolist(), total_tokens

    def predict_batched(self, texts: list[str], batch_size: int = 32) -> tuple[list[list[float]], int]:
        """
        Process texts in batches to avoid OOM errors.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (default: 32)
            
        Returns:
            Tuple of (List of embedding vectors, total_tokens)
        """
        all_embeddings = []
        total_tokens_count = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings, tokens = self.predict(batch)
            all_embeddings.extend(embeddings)
            total_tokens_count += tokens
        return all_embeddings, total_tokens_count

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask

    def normalize(self, v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norm, a_min=1e-9, a_max=None)
