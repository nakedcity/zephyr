# Zephyr — fast, pure-ONNX embeddings server

Zephyr delivers OpenAI-compatible embedding and model endpoints without the weight of PyTorch. It downloads ONNX models from Hugging Face, serves them via FastAPI, and runs on GPU or CPU with optional quantization—ideal when you want small, fast inference.

## Why you’ll like it (quick pitch)
- **OpenAI-compatible surface:** `/v1/embeddings`, `/v1/models`, retrieve, and delete endpoints; Bearer auth mirrors OpenAI keys.
- **Zero PyTorch bloat:** pure ONNX Runtime, quick cold starts.
- **Pluggable performance:** switch GPU/CPU per model; opt-in quantization for tight footprints.
- **Smart caching:** configurable LRU of loaded models plus preload list for warm starts.
- **DX-minded:** tiny dependency set, pytest suite, and straightforward config via `config/config.yaml`.

## Short overview
- FastAPI app boots with a `ModelCache` that preloads any models listed in `preload`.
- Requests authenticate via `Authorization: Bearer <OPENAI_API_KEY>`.
- On-demand loads pull `model.onnx` + tokenizer from Hugging Face Hub, optionally quantize, then serve through ONNX Runtime.
- LRU cache manages memory and tracks per-model `created` timestamps; delete endpoint unloads models.

## Architecture (Multi-Process GPU Separation)
Zephyr uses a multi-process architecture to isolate GPU engines. This is critical because `onnxruntime-gpu` (CUDA) and `onnxruntime-migraphx` (ROCm) components often have conflicting shared library requirements and cannot easily coexist in the same Python process.

```mermaid
flowchart TD
    Client -->|HTTP /v1/*| Gateway[FastAPI Gateway]
    Gateway -->|Forward| ProcessManager
    ProcessManager -->|Spawn| CUDA[CUDA Worker (.venv-cuda)]
    ProcessManager -->|Spawn| MIGraphX[MIGraphX Worker (.venv-migraphx)]
    ProcessManager -->|Spawn| ROCm[ROCm Worker (.venv-rocm)]
    ProcessManager -->|Spawn| CPU[CPU Worker (.venv-cpu)]
    
    CUDA -->|Inference| Model1[Embedding Model A]
    MIGraphX -->|Inference| Model2[Embedding Model B]
    ROCm -->|Inference| Model3[Embedding Model C]
```

## Installation & Setup
Zephyr isolates environments automatically using provided scripts.

**1. Create Environments:**
Run the install script to generate dedicated virtual environments for CPU, CUDA, MIGraphX and ROCm.
```bash
./install.sh
```

**2. Configure Models:**
Assign each model to a specific engine in `config/config.yaml`.
```yaml
models:
  bge-small-en-v1.5:
    engine: "migraphx" # "cuda", "migraphx", "rocm", or "cpu"
```

**3. Run:**
Start the main server. The Gateway will automatically spawn the necessary worker processes based on your config.

> [!TIP]
> Use the **CPU environment** (`.venv-cpu`) to run the Gateway server. It is lightweight and has all necessary dependencies (including `onnxruntime` for quantization if needed).

```bash
# Export your API key
export OPENAI_API_KEY=your_key

# Run the server using the helper script
./run.sh

# Or manually:
# source .venv-cpu/bin/activate
# fastapi run server/main.py --port 8080
```

Key configs live in `config/config.yaml`—set per-model `repo`, `engine` (cuda/migraphx/rocm/cpu), `quantize`, and `owner`; adjust `cache` and `preload` to fit your deployment.

## Contributing

## Contributing
- Fork the repo and work on a branch in your fork.
- Open a pull request to `main`; CI will run and we review/merge.
