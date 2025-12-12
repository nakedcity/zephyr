# Zephyr — fast, pure-ONNX embeddings server

Zephyr delivers OpenAI-compatible embedding and model endpoints without the weight of PyTorch. It downloads ONNX models from Hugging Face, serves them via FastAPI, and runs on GPU or CPU with optional quantization—ideal when you want small, fast, and container-friendly inference.

## Why you’ll like it (quick pitch)
- **OpenAI-compatible surface:** `/v1/embeddings`, `/v1/models`, retrieve, and delete endpoints; Bearer auth mirrors OpenAI keys.
- **Zero PyTorch bloat:** pure ONNX Runtime, slim container, quick cold starts.
- **Pluggable performance:** switch GPU/CPU per model; opt-in quantization for tight footprints.
- **Smart caching:** configurable LRU of loaded models plus preload list for warm starts.
- **DX-minded:** tiny dependency set, pytest suite, and straightforward config via `config/config.yaml`.

## Short overview
- FastAPI app boots with a `ModelCache` that preloads any models listed in `preload`.
- Requests authenticate via `Authorization: Bearer <OPENAI_API_KEY>`.
- On-demand loads pull `model.onnx` + tokenizer from Hugging Face Hub, optionally quantize, then serve through ONNX Runtime.
- LRU cache manages memory and tracks per-model `created` timestamps; delete endpoint unloads models.

## Architecture (Multi-Process GPU Separation)
Zephyr uses a multi-process architecture to isolate GPU engines. This is critical because `onnxruntime-gpu` (CUDA) and `onnxruntime-rocm` components often have conflicting shared library requirements and cannot easily coexist in the same Python process.

```mermaid
flowchart TD
    Client -->|HTTP /v1/*| Gateway[FastAPI Gateway]
    Gateway -->|Forward| ProcessManager
    ProcessManager -->|Spawn| CUDA[CUDA Worker (.venv-cuda)]
    ProcessManager -->|Spawn| ROCm[ROCm Worker (.venv-rocm)]
    ProcessManager -->|Spawn| CPU[CPU Worker (.venv-cpu)]
    
    CUDA -->|Inference| Model1[Embedding Model A]
    ROCm -->|Inference| Model2[Embedding Model B]
```

## Installation & Setup
Zephyr isolates environments automatically using provided scripts.

**1. Create Environments:**
Run the install script to generate dedicated virtual environments for CPU, CUDA, and ROCm.
```bash
./install.sh
```

**2. Configure Models:**
Assign each model to a specific engine in `config/config.yaml`.
```yaml
models:
  bge-small-en-v1.5:
    engine: "rocm" # "cuda", "rocm", or "cpu"
```

**3. Run:**
Start the main server. The Gateway will automatically spawn the necessary worker processes based on your config.
```bash
# You can run the server using your system python or any venv
python server/main.py
```

## Run it
1) `export OPENAI_API_KEY=your_key`
2) `uvicorn server.main:app --reload` (or use the Dockerfile)

Key configs live in `config/config.yaml`—set per-model `repo`, `device`, `quantize`, and `owner`; adjust `cache` and `preload` to fit your deployment. 

## Contributing
- Fork the repo and work on a branch in your fork.
- Open a pull request to `main`; CI will run and we review/merge.
