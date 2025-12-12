from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import numpy as np
import os

@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def auth_headers():
    return {"Authorization": "Bearer test-key"}

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_list_models():
    with patch('server.main.ModelCache') as MockCache:
        mock_instance = MockCache.return_value
        with TestClient(app) as client:
            response = client.get("/v1/models", headers=auth_headers())
            assert response.status_code == 200
            data = response.json()['data']
            assert len(data) > 0
            ids = [m['id'] for m in data]
            assert 'bge-small-en-v1.5' in ids

def test_create_embedding():
    mock_embedder = MagicMock()
    mock_embedder.predict_batched = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    
    # ModelCache now returns RemoteEmbedder
    mock_cache = MagicMock()
    mock_cache.get_model.return_value = mock_embedder
    
    with patch('server.main.ModelCache') as MockCache:
        MockCache.return_value = mock_cache
        
        with TestClient(app) as client:
            response = client.post("/v1/embeddings",
                                   headers=auth_headers(),
                                   json={
                                       "input": "hello",
                                       "model": "bge-small-en-v1.5"
                                   })
            
            assert response.status_code == 200
            json_resp = response.json()
            assert json_resp['data'][0]['embedding'] == [0.1, 0.2, 0.3]

def test_quantization_workflow():
    # ModelCache -> ProcessManager -> start_worker
    # We test that quantization happens before worker start
    with patch('embeddings.model_cache.hf_hub_download') as mock_download, \
         patch('embeddings.model_cache.ProcessManager') as MockPM, \
         patch('embeddings.model_cache.RemoteEmbedder') as MockRemote, \
         patch('embeddings.model_cache.quantize_model') as mock_quantize, \
         patch('os.path.exists') as mock_exists:
        
        mock_download.return_value = "/tmp/model.onnx"
        # Say quantized model does NOT exist yet
        mock_exists.side_effect = lambda p: p == "/tmp/model.onnx"
        
        MockPM.return_value.start_worker_for_model.return_value = 5001
        
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
                    "engine": "cpu"
                }
            },
            "workers": {"cpu": {"port": 5003}}
        })
        
        cache = ModelCache(conf)
        cache.get_model("test-quant")
        
        mock_quantize.assert_called_once()
        # Verify worker started with quantized path
        args, kwargs = MockPM.return_value.start_worker_for_model.call_args
        assert "model_quantized.onnx" in args[1]

def test_engine_selection_cuda():
    # Test that ModelCache reads 'engine' correctly and ProcessManager is initialized
    # Actual mapping logic is in ProcessManager, so we should test ProcessManager separately ideally.
    # But here we verify ModelCache integration.
    
    with patch('embeddings.model_cache.hf_hub_download') as mock_download, \
         patch('embeddings.model_cache.ProcessManager') as MockPM, \
         patch('embeddings.model_cache.RemoteEmbedder'):
        
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
                    "engine": "cuda"
                }
            },
            "workers": {"cuda": {"port": 5001}}
        })
        
        cache = ModelCache(conf)
        cache.get_model("test-gpu")
        
        MockPM.return_value.start_worker_for_model.assert_called()


def test_process_manager_mapping():
    # Direct test of ProcessManager mapping logic
    from server.process_manager import ProcessManager
    from omegaconf import OmegaConf
    from unittest.mock import patch
    
    conf = OmegaConf.create({
        "models": {
            "m1": {"engine": "cuda", "max_tokens": 128},
            "m2": {"engine": "migraphx", "max_tokens": 128},
            "m3": {"engine": "cpu",  "max_tokens": 64},
            "m4": {"engine": "rocm", "max_tokens": 128}
        },
        "workers": {
            "cuda": {"port": 5001},
            "migraphx": {"port": 5002},
            "rocm": {"port": 5004},
            "cpu":  {"port": 5003}
        }
    })
    
    pm = ProcessManager(conf)
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('server.process_manager.is_port_in_use', return_value=False), \
         patch('server.process_manager.Path.exists', return_value=True), \
         patch.object(pm, '_wait_for_health'):
        
        # Test CUDA
        pm.start_worker_for_model("m1", "model.path", "tok.path")
        kwargs_cuda = mock_popen.call_args[1]
        env_cuda = kwargs_cuda['env']
        assert env_cuda["ZEPHYR_DEVICE"] == "gpu"
        assert env_cuda["ZEPHYR_PROVIDER"] == "cuda"
        
        # Test MIGraphX
        pm.start_worker_for_model("m2", "/path/m2", "/path/tok2")
        kwargs_migraphx = mock_popen.call_args[1]
        env_migraphx = kwargs_migraphx['env']
        assert env_migraphx["ZEPHYR_DEVICE"] == "gpu"
        assert env_migraphx["ZEPHYR_PROVIDER"] == "migraphx"
        
        # Test ROCm
        pm.start_worker_for_model("m4", "/path/m4", "/path/tok4")
        kwargs_rocm = mock_popen.call_args[1]
        env_rocm = kwargs_rocm['env']
        assert env_rocm["ZEPHYR_DEVICE"] == "gpu"
        assert env_rocm["ZEPHYR_PROVIDER"] == "rocm"
        
        # Test CPU
        pm.start_worker_for_model("m3", "model.path", "tok.path")
        kwargs_cpu = mock_popen.call_args[1]
        env_cpu = kwargs_cpu['env']
        assert env_cpu["ZEPHYR_DEVICE"] == "cpu"
        # Provider might not be set or set to None/empty
        assert "ZEPHYR_PROVIDER" not in env_cpu or env_cpu["ZEPHYR_PROVIDER"] == "none"


def test_retrieve_model_loads_and_returns_metadata():
    with patch('server.main.ModelCache') as MockCache:
        mock_cache = MockCache.return_value
        mock_cache.get_model.return_value = MagicMock()
        mock_cache.get_created_timestamp.return_value = 1700000000

        with TestClient(app) as client:
            resp = client.get("/v1/models/bge-small-en-v1.5", headers=auth_headers())
            assert resp.status_code == 200
            data = resp.json()
            assert data['id'] == 'bge-small-en-v1.5'
            assert data['object'] == 'model'
            assert data['owned_by'] == 'Xenova'
            assert data['created'] == 1700000000
        mock_cache.get_model.assert_called_with('bge-small-en-v1.5')

def test_delete_model_unloads():
    with patch('server.main.ModelCache') as MockCache:
        mock_cache = MockCache.return_value
        mock_cache.unload_model.return_value = True

        with TestClient(app) as client:
            resp = client.delete("/v1/models/all-MiniLM-L6-v2", headers=auth_headers())
            assert resp.status_code == 200
            data = resp.json()
            assert data['id'] == 'all-MiniLM-L6-v2'
            assert data['object'] == 'model'
            assert data['deleted'] is True
        mock_cache.unload_model.assert_called_with('all-MiniLM-L6-v2')

def test_retrieve_unknown_model_returns_404():
    with TestClient(app) as client:
        resp = client.get("/v1/models/does-not-exist", headers=auth_headers())
        assert resp.status_code == 404
        assert "not found" in resp.json()['detail']

def test_missing_auth_gets_401():
    from omegaconf import OmegaConf
    mock_conf = OmegaConf.create({
        'authorization': {'enabled': True, 'token_env_var': 'OPENAI_API_KEY'},
        'cache': {'directory': '/tmp/cache', 'max_loaded_models': 1},
        'models': {},
        'preload': [],
        'workers': {}
    })
    
    with patch('server.main.config', mock_conf):
        with TestClient(app) as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 401
            assert "Authorization" in resp.json()["detail"]
