import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import os

class ONNXEmbedder:
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 512, device: str = "cpu"):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Enable truncation and padding
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length)

        # Load ONNX model
        if device == "gpu":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        print(f"Requesting load on {device} with providers: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Verify actual providers
        active_providers = self.session.get_providers()
        print(f"Model loaded. Active providers: {active_providers}")
        
        if device == "gpu" and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                f"GPU requested but CUDAExecutionProvider not available. "
                f"Active providers: {active_providers}. "
                "Check if onnxruntime-gpu is installed and CUDA is available."
            )
        
    def predict(self, texts: list[str]) -> list[list[float]]:
        # Tokenize
        encoded = self.tokenizer.encode_batch(texts)
        
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
        
        return embeddings.tolist()

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
