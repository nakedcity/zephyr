import subprocess
import sys
import time
import os
import signal
import socket
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

class ProcessManager:
    def __init__(self, config):
        self.config = config
        self.processes = {} # model_id -> subprocess.Popen

    def start_worker_for_model(self, model_id: str, model_path: str, tokenizer_path: str):
        if model_id in self.processes:
            return self.processes[model_id] # Already running (maybe shared worker?)
            
        model_conf = self.config.models[model_id]
        # engine: cuda, rocm, or cpu
        engine = getattr(model_conf, 'engine', 'cpu')
        
        # Determine worker config based on engine
        # Config schema:
        # workers:
        #   cuda: { port: 5001 }
        #   rocm: { port: 5002 }
        #   cpu:  { port: 5003 }
        
        worker_conf = self.config.workers.get(engine)
        if not worker_conf:
             raise ValueError(f"No worker configuration found for engine '{engine}'")

        port = worker_conf.port
        
        # Determine python interpreter path and worker args
        # engine=cuda -> .venv-cuda, device=gpu, provider=cuda
        # engine=rocm -> .venv-rocm, device=gpu, provider=rocm
        # engine=cpu  -> .venv-cpu,  device=cpu,  provider=None
        
        if engine == "cuda":
            venv_name = ".venv-cuda"
            provider = "cuda"
            device_arg = "gpu"
        elif engine == "rocm":
            venv_name = ".venv-rocm"
            provider = "rocm"
            device_arg = "gpu"
        else:
            venv_name = ".venv-cpu"
            provider = "none" # Argparser expects string
            device_arg = "cpu"

        # Construct path to python
        project_root = Path(__file__).resolve().parent.parent
        python_exe = project_root / venv_name / "bin" / "python"
        
        if not python_exe.exists():
             raise RuntimeError(f"Python interpreter not found at {python_exe}. Run ./install.sh first.")

        # Prepare environment variables for worker
        env = os.environ.copy()
        env.update({
            "ZEPHYR_MODEL_PATH": str(model_path),
            "ZEPHYR_TOKENIZER_PATH": str(tokenizer_path),
            "ZEPHYR_MAX_LENGTH": str(model_conf.max_tokens),
            "ZEPHYR_DEVICE": device_arg,
        })
        if provider != "none":
            env["ZEPHYR_PROVIDER"] = provider

        # Use 'fastapi run' to start the worker
        # We run it via 'python -m fastapi run' to ensure we use the venv's fastapi
        cmd = [
            str(python_exe),
            "-m", "fastapi", "run",
            str(project_root / "server" / "worker.py"),
            "--port", str(port)
        ]
        
        logger.info(f"Starting worker for {model_id} on port {port} with {venv_name}...")
        
        # Check if port is in use
        if is_port_in_use(port):
             logger.warning(f"Port {port} is already in use!")

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        self.processes[model_id] = proc
        
        # Wait for health check
        self._wait_for_health(port)
        return port

    def _wait_for_health(self, port: int, timeout=30):
        # We can implement a retry loop here using requests
        import requests
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Worker on port {port} failed to start within {timeout}s")

    def stop_all(self):
        for model_id, proc in self.processes.items():
            logger.info(f"Stopping worker for {model_id}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.processes.clear()
