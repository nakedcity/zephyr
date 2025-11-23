from omegaconf import OmegaConf
import os

def load_config(config_path: str = "config.yaml"):
    """
    Load configuration from a YAML file using OmegaConf.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    conf = OmegaConf.load(config_path)
    return conf
