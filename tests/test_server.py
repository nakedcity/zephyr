from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_list_models():
    with patch('server.main.ModelCache') as MockCache:
        mock_instance = MockCache.return_value
        with TestClient(app) as client:
            response = client.get("/v1/models")
            assert response.status_code == 200
            data = response.json()['data']
            assert len(data) > 0
            ids = [m['id'] for m in data]
            assert 'bge-small-en-v1.5' in ids

def test_create_embedding():
    mock_embedder = MagicMock()
    mock_embedder.predict.return_value = [[0.1, 0.2, 0.3]]
    
    mock_cache = MagicMock()
    mock_cache.get_model.return_value = mock_embedder
    
    with patch('server.main.ModelCache') as MockCache:
        MockCache.return_value = mock_cache
        
        with TestClient(app) as client:
            response = client.post("/v1/embeddings", json={
                "input": "hello",
                "model": "bge-small-en-v1.5"
            })
            
            assert response.status_code == 200
            json_resp = response.json()
            assert json_resp['data'][0]['embedding'] == [0.1, 0.2, 0.3]

def test_quantization_workflow():
    with patch('embeddings.model_cache.hf_hub_download') as mock_download, \
         patch('embeddings.model_cache.ONNXEmbedder') as MockEmbedder, \
         patch('embeddings.model_cache.quantize_model') as mock_quantize, \
         patch('os.path.exists') as mock_exists:
        
        mock_download.return_value = "/tmp/model.onnx"
        mock_exists.side_effect = lambda p: p == "/tmp/model.onnx"
        
        from embeddings.model_cache import ModelCache
        from omegaconf import OmegaConf
        
        conf = OmegaConf.create({
            "cache": {"directory": "/tmp", "max_loaded_models": 1},
            "models": {
                "test-quant": {
                    "repo": "test/repo",
                    "dimension": 384,
                    "max_tokens": 512,
                    "quantize": True,
                    "device": "cpu"
                }
            }
        })
        
        cache = ModelCache(conf)
        cache.get_model("test-quant")
        
        mock_quantize.assert_called_once()
        args, kwargs = MockEmbedder.call_args
        assert "model_quantized.onnx" in args[0]
        assert kwargs['device'] == 'cpu'

def test_device_selection_gpu_success():
    with patch('embeddings.model_cache.hf_hub_download') as mock_download, \
         patch('embeddings.model_cache.ONNXEmbedder') as MockEmbedder:
        
        mock_download.return_value = "/tmp/model.onnx"
        
        from embeddings.model_cache import ModelCache
        from omegaconf import OmegaConf
        
        conf = OmegaConf.create({
            "cache": {"directory": "/tmp", "max_loaded_models": 1},
            "models": {
                "test-gpu": {
                    "repo": "test/repo",
                    "dimension": 384,
                    "max_tokens": 512,
                    "quantize": False,
                    "device": "gpu"
                }
            }
        })
        
        cache = ModelCache(conf)
        cache.get_model("test-gpu")
        
        args, kwargs = MockEmbedder.call_args
        assert kwargs['device'] == 'gpu'

def test_device_selection_gpu_fail():
    # Test that RuntimeError is raised if CUDA is missing
    # We need to mock ONNXEmbedder to simulate the initialization failure
    # But wait, ONNXEmbedder is what we are testing. We should mock ort.InferenceSession
    
    with patch('embeddings.embedder.ort.InferenceSession') as MockSession, \
         patch('embeddings.embedder.Tokenizer'):
        
        # Simulate CPU fallback
        mock_session = MockSession.return_value
        mock_session.get_providers.return_value = ['CPUExecutionProvider']
        
        from embeddings.embedder import ONNXEmbedder
        
        with pytest.raises(RuntimeError) as excinfo:
            ONNXEmbedder("model.onnx", "tokenizer.json", device="gpu")
        
        assert "GPU requested but CUDAExecutionProvider not available" in str(excinfo.value)
