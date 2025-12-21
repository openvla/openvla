"""
OpenVLA Edge Optimization Package

Optimizes OpenVLA 7B for Jetson Orin deployment through:
- AWQ INT4 quantization with robot-data calibration
- Custom action-only logits processor
- High-performance inference pipeline
"""

__version__ = "0.1.0"
__author__ = "OpenVLA Edge Optimization"

from .action_logits_processor import ActionOnlyLogitsProcessor
from .action_decoder import ActionDecoder

__all__ = [
    "ActionOnlyLogitsProcessor",
    "ActionDecoder",
]
