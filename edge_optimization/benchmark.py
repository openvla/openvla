"""
Benchmark Script for OpenVLA Edge Optimization

Measures inference latency and throughput for different model configurations.

Usage:
    python benchmark.py --model ./openvla-7b-awq --iterations 20
    python benchmark.py --model openvla/openvla-7b --quantized false
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_optimization.optimized_inference import OpenVLAInference


def run_benchmark(
    model_path: str,
    quantized: bool = True,
    iterations: int = 20,
    warmup: int = 3,
    output_file: str = None
) -> dict:
    """
    Run inference benchmark and report results.
    
    Args:
        model_path: Path to model
        quantized: Whether model is AWQ quantized
        iterations: Number of benchmark iterations
        warmup: Warmup iterations (not counted)
        output_file: Optional JSON file to save results
        
    Returns:
        Benchmark results dictionary
    """
    print("=" * 60)
    print("OpenVLA Edge Optimization Benchmark")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Quantized: {quantized}")
    print(f"Iterations: {iterations}")
    print("=" * 60)
    
    # Initialize model
    print("\nInitializing model...")
    model = OpenVLAInference(
        model_path=model_path,
        quantized=quantized,
        use_flash_attention=True
    )
    model.load()
    
    # Get memory usage
    memory = model.get_memory_usage()
    print(f"\nGPU Memory:")
    if memory.get("available", True):
        print(f"  Allocated: {memory['allocated_mb']:.1f} MB")
        print(f"  Reserved: {memory['reserved_mb']:.1f} MB")
    else:
        print("  CUDA not available")
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = model.benchmark(
        num_iterations=iterations,
        warmup=warmup
    )
    
    # Add metadata
    results["model_path"] = model_path
    results["quantized"] = quantized
    results["memory"] = memory
    results["timestamp"] = datetime.now().isoformat()
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean Latency: {results['mean_ms']:.2f} ms")
    print(f"Std Dev: {results['std_ms']:.2f} ms")
    print(f"Min: {results['min_ms']:.2f} ms")
    print(f"Max: {results['max_ms']:.2f} ms")
    print(f"P50 (Median): {results['p50_ms']:.2f} ms")
    print(f"P95: {results['p95_ms']:.2f} ms")
    print(f"P99: {results['p99_ms']:.2f} ms")
    print("-" * 60)
    print(f"Mean FPS: {results['mean_fps']:.2f} Hz")
    print("=" * 60)
    
    # Check targets
    target_hz = 5.0
    target_ms = 1000 / target_hz
    
    if results['mean_ms'] <= target_ms:
        print(f"✓ PASS: Meets {target_hz} Hz target (>{target_ms:.0f}ms)")
    else:
        print(f"✗ FAIL: Does not meet {target_hz} Hz target")
        print(f"  Need {results['mean_ms'] - target_ms:.1f}ms improvement")
    
    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
    
    return results


def compare_models(
    original_path: str,
    quantized_path: str,
    iterations: int = 10
):
    """
    Compare original vs quantized model performance.
    
    Args:
        original_path: Path to original FP16 model
        quantized_path: Path to AWQ quantized model
        iterations: Benchmark iterations per model
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Benchmark original
    print("\n[1/2] Benchmarking ORIGINAL model...")
    original_results = run_benchmark(
        model_path=original_path,
        quantized=False,
        iterations=iterations
    )
    
    # Clear GPU memory
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark quantized
    print("\n[2/2] Benchmarking QUANTIZED model...")
    quantized_results = run_benchmark(
        model_path=quantized_path,
        quantized=True,
        iterations=iterations
    )
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    speedup = original_results['mean_ms'] / quantized_results['mean_ms']
    memory_reduction = (
        original_results['memory'].get('allocated_mb', 0) - 
        quantized_results['memory'].get('allocated_mb', 0)
    )
    
    print(f"Original Latency: {original_results['mean_ms']:.2f} ms")
    print(f"Quantized Latency: {quantized_results['mean_ms']:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Memory Savings: {memory_reduction:.0f} MB")
    print(f"Original FPS: {original_results['mean_fps']:.2f} Hz")
    print(f"Quantized FPS: {quantized_results['mean_fps']:.2f} Hz")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenVLA inference performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (HuggingFace or local)"
    )
    parser.add_argument(
        "--quantized",
        type=str,
        default="true",
        help="Whether model is AWQ quantized (true/false)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to original model for comparison"
    )
    
    args = parser.parse_args()
    
    quantized = args.quantized.lower() in ("true", "1", "yes")
    
    if args.compare:
        compare_models(
            original_path=args.compare,
            quantized_path=args.model,
            iterations=args.iterations
        )
    else:
        run_benchmark(
            model_path=args.model,
            quantized=quantized,
            iterations=args.iterations,
            warmup=args.warmup,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
