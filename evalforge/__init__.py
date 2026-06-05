"""
EvalForge: Local-first, GPU-accelerated LLM evaluation & benchmarking toolkit.
"""

__version__ = "0.3.1"
__author__ = "Nhaked-try"

from evalforge.evaluator import Evaluator
from evalforge.arena import Arena
from evalforge.reporter import Reporter

__all__ = ["Evaluator", "Arena", "Reporter"]
