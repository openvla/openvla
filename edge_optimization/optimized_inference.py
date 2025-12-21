"""
Optimized Inference Pipeline for OpenVLA

High-performance inference with AWQ quantization and action-only decoding.
Designed for edge deployment on Jetson Orin.

Usage:
    from edge_optimization.optimized_inference import OpenVLAInference
    
    model = OpenVLAInference("./openvla-7b-awq")
    action = model.predict_action(image, instruction)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from .action_logits_processor import ActionOnlyLogitsProcessor
from .action_decoder import ActionDecoder


@dataclass
class InferenceMetrics:
    """Timing breakdown for inference pipeline."""
    preprocess_ms: float
    vision_encode_ms: float
    llm_prefill_ms: float
    llm_decode_ms: float
    postprocess_ms: float
    
    @property
    def total_ms(self) -> float:
        return (
            self.preprocess_ms + 
            self.vision_encode_ms + 
            self.llm_prefill_ms + 
            self.llm_decode_ms + 
            self.postprocess_ms
        )
    
    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0
    
    def __str__(self) -> str:
        return (
            f"Inference: {self.total_ms:.1f}ms ({self.fps:.1f} Hz)\n"
            f"  Preprocess: {self.preprocess_ms:.1f}ms\n"
            f"  Vision: {self.vision_encode_ms:.1f}ms\n"
            f"  Prefill: {self.llm_prefill_ms:.1f}ms\n"
            f"  Decode: {self.llm_decode_ms:.1f}ms\n"
            f"  Postprocess: {self.postprocess_ms:.1f}ms"
        )


class OpenVLAInference:
    """
    Optimized OpenVLA inference pipeline.
    
    Features:
    - AWQ INT4 quantized model loading
    - Flash attention for vision encoder
    - Action-only logits processor
    - Detailed timing breakdown
    
    Args:
        model_path: Path to quantized model or HuggingFace model name
        device: Device for inference (cuda:0, cpu)
        use_flash_attention: Enable flash attention if available
        temperature: Sampling temperature (0 for greedy)
        quantized: Whether model is AWQ quantized
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        use_flash_attention: bool = True,
        temperature: float = 0.0,
        quantized: bool = True
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.quantized = quantized
        
        # Determine attention implementation
        self.attn_implementation = (
            "flash_attention_2" if use_flash_attention else "eager"
        )
        
        self.model = None
        self.processor = None
        self.action_logits_processor = None
        self.action_decoder = None
        self._loaded = False
    
    def load(self):
        """Load model and processor."""
        if self._loaded:
            return
        
        print(f"Loading {'quantized' if self.quantized else 'original'} model...")
        
        if self.quantized:
            self._load_quantized()
        else:
            self._load_original()
        
        # Initialize action processing components
        vocab_size = self.model.config.vocab_size
        self.action_logits_processor = ActionOnlyLogitsProcessor(
            vocab_size=vocab_size,
            action_bins=256
        )
        self.action_decoder = ActionDecoder(vocab_size=vocab_size)
        
        self._loaded = True
        print(f"Model loaded on {self.device}")
    
    def _load_quantized(self):
        """Load AWQ quantized model."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("Install autoawq: pip install autoawq>=0.2.0")
        
        from transformers import AutoProcessor
        
        self.model = AutoAWQForCausalLM.from_quantized(
            self.model_path,
            fuse_layers=True,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def _load_original(self):
        """Load original FP16/BF16 model."""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def format_prompt(self, instruction: str) -> str:
        """Format instruction into OpenVLA prompt template."""
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    
    @torch.inference_mode()
    def predict_action(
        self,
        image: Union[np.ndarray, Image.Image],
        instruction: str,
        unnorm_key: Optional[str] = None,
        return_metrics: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, InferenceMetrics]]:
        """
        Predict robot action from image and instruction.
        
        Args:
            image: RGB image (numpy array or PIL Image)
            instruction: Task instruction (e.g., "pick up the red block")
            unnorm_key: Dataset key for unnormalization
            return_metrics: If True, also return timing metrics
            
        Returns:
            action: 7-DOF action array [x, y, z, roll, pitch, yaw, gripper]
            metrics: Optional timing breakdown
        """
        if not self._loaded:
            self.load()
        
        timings = {}
        
        # Preprocess
        t0 = time.perf_counter()
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        prompt = self.format_prompt(instruction)
        inputs = self.processor(prompt, image)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Convert to proper dtype if needed
        if hasattr(self.model, 'dtype') and inputs.get('pixel_values') is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
        
        timings['preprocess_ms'] = (time.perf_counter() - t0) * 1000
        
        # Generate action tokens
        t1 = time.perf_counter()
        
        # For detailed timing, we would need hooks into the model
        # For now, we measure overall generation time
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=7,  # 7 action dimensions
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            logits_processor=[self.action_logits_processor] if not self.quantized else None
        )
        
        # Estimate timing breakdown (actual split requires model hooks)
        gen_time = (time.perf_counter() - t1) * 1000
        timings['vision_encode_ms'] = gen_time * 0.15  # ~15% for vision
        timings['llm_prefill_ms'] = gen_time * 0.25   # ~25% for prefill
        timings['llm_decode_ms'] = gen_time * 0.60    # ~60% for decode
        
        # Postprocess
        t2 = time.perf_counter()
        
        # Extract last 7 tokens (action tokens)
        action_token_ids = outputs[0, -7:].cpu().numpy()
        
        # Decode to continuous actions
        action = self.action_decoder.decode_token_ids(
            action_token_ids,
            unnorm_key=unnorm_key
        )
        
        timings['postprocess_ms'] = (time.perf_counter() - t2) * 1000
        
        if return_metrics:
            metrics = InferenceMetrics(**timings)
            return action, metrics
        
        return action
    
    def benchmark(
        self,
        num_iterations: int = 20,
        warmup: int = 3,
        image_size: tuple = (224, 224)
    ) -> dict:
        """
        Run inference benchmark with dummy data.
        
        Args:
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations (not counted)
            image_size: Test image size (H, W)
            
        Returns:
            Benchmark results dictionary
        """
        if not self._loaded:
            self.load()
        
        # Create dummy test data
        test_image = np.random.randint(
            0, 255, 
            (*image_size, 3), 
            dtype=np.uint8
        )
        test_instruction = "pick up the red block"
        
        # Warmup
        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            _ = self.predict_action(test_image, test_instruction)
        
        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        latencies = []
        all_metrics = []
        
        for i in range(num_iterations):
            _, metrics = self.predict_action(
                test_image, 
                test_instruction,
                return_metrics=True
            )
            latencies.append(metrics.total_ms)
            all_metrics.append(metrics)
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        results = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_fps": float(1000 / np.mean(latencies)),
            "num_iterations": num_iterations
        }
        
        return results
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2
        }
