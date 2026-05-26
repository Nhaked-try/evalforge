"""Shared utilities for EvalForge."""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict
import os


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml required: pip install pyyaml")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_cache_key(prompt_text: str, model_name: str) -> str:
    """Compute a cache key for a prompt+model combination."""
    raw = f"{model_name}:{prompt_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def get_results_dir() -> Path:
    """Get results directory from env or default."""
    return Path(os.getenv("EVALFORGE_RESULTS_DIR", "./results"))
