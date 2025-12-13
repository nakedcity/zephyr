
import subprocess
import sys
import time
import os
import signal
import socket
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

class ProcessManager:
    def __init__(self, config):
        self.config = config
        self.processes = {} # model_id -> subprocess.Popen

    def start_worker_for_model(self, model_id: str, model_path: str, tokenizer_path: str) -> int:
        if model_id in self.processes:
            proc, port = self.processes[model_id]
            if proc.poll() is None:
                return port # Return the port of the running worker
            # If process is dead, clean up and restart
            self.stop_worker(model_id) 
            
        model_conf = self.config.models[model_id]
        # engine: cuda, migraphx, or cpu
        engine = getattr(model_conf, 'engine', 'cpu')
        
        # Determine worker config based on engine
        # Config schema:
        # workers:
        #   cuda:     { port: 5001 }
        #   migraphx: { port: 5002 }
        #   cpu:      { port: 5003 }
        
        # Find a free port
        port = self._find_free_port()
        
        # Determine python interpreter path and worker args
        # engine=cuda     -> .venv-cuda,     device=gpu, provider=cuda
        # engine=migraphx -> .venv-migraphx, device=gpu, provider=migraphx
        # engine=cpu      -> .venv-cpu,      device=cpu, provider=None
        
        if engine == "cuda":
            venv_name = ".venv-cuda"
            provider = "cuda"
            device_arg = "gpu"
        elif engine == "migraphx":
            venv_name = ".venv-migraphx"
            provider = "migraphx"
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
        
        # Batch size config
        batch_size = getattr(model_conf, 'batch_size', 32)

        env.update({
            "ZEPHYR_MODEL_PATH": str(model_path),
            "ZEPHYR_TOKENIZER_PATH": str(tokenizer_path),
            "ZEPHYR_MAX_LENGTH": str(model_conf.max_tokens),
            "ZEPHYR_DEVICE": device_arg,
            "ZEPHYR_BATCH_SIZE": str(batch_size),
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
            stderr=subprocess.PIPE, # Capture stderr for filtering
            text=True # Text mode for line buffering
        )
        
        # Background thread to filter stderr spam
        def filter_stderr(pipe):
            for line in pipe:
                if "MIGraphX: param type mismatch" in line:
                    continue
                sys.stderr.write(line)
                sys.stderr.flush()
                
        t = threading.Thread(target=filter_stderr, args=(proc.stderr,))
        t.daemon = True
        t.start()
        
        self.processes[model_id] = (proc, port)
        
        # Wait for health check
        self._wait_for_health(port, timeout=300)
        return port

    def _wait_for_health(self, port: int, timeout=300):
        # We can implement a retry loop here using httpx
        import httpx
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Worker on port {port} failed to start within {timeout}s")

    def _find_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    def stop_worker(self, model_id: str) -> bool:
        entry = self.processes.pop(model_id, None)
        if not entry:
            return False
        
        proc, _ = entry
        logger.info(f"Stopping worker for {model_id}...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    def stop_all(self):
        # Copy keys to avoid mutation during iteration
        for model_id in list(self.processes.keys()):
            self.stop_worker(model_id)
