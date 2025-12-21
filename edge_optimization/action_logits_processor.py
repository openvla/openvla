"""
Action Logits Processor for OpenVLA

Forces the model to ONLY output action tokens (last 256 tokens in vocabulary).
This prevents the model from generating text instead of robot actions.

Usage:
    from edge_optimization import ActionOnlyLogitsProcessor
    
    processor = ActionOnlyLogitsProcessor(vocab_size=32000, action_bins=256)
    # Use with transformers generate() or vLLM sampling
"""

import torch
from typing import Optional


class ActionOnlyLogitsProcessor:
    """
    A logits processor that constrains LLM output to action tokens only.
    
    OpenVLA maps continuous robot actions (7-DOF) to discrete tokens using
    the LAST 256 tokens of the vocabulary. This processor masks all other
    tokens with -inf, ensuring the model only outputs valid action tokens.
    
    Mathematical operation:
        For logits L ∈ R^{batch × vocab_size}:
        L_i = L_i if i ∈ [vocab_size - 256, vocab_size)
        L_i = -∞  otherwise
    
    Args:
        vocab_size: Total vocabulary size of the model (e.g., 32000 for Llama)
        action_bins: Number of action bins (default: 256)
        device: Device to create mask on (default: auto-detect from input)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        action_bins: int = 256,
        device: Optional[str] = None
    ):
        self.vocab_size = vocab_size
        self.action_bins = action_bins
        self.action_start_idx = vocab_size - action_bins
        self.action_end_idx = vocab_size
        self.device = device
        
        # Pre-compute mask for efficiency (created on first call if device not set)
        self._cached_mask: Optional[torch.Tensor] = None
    
    def _get_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Get or create the logits mask with proper device/dtype."""
        if self._cached_mask is None or self._cached_mask.device != scores.device:
            mask = torch.full(
                (self.vocab_size,), 
                float('-inf'), 
                device=scores.device, 
                dtype=scores.dtype
            )
            # Unmask action tokens (set to 0 so adding doesn't change scores)
            mask[self.action_start_idx:self.action_end_idx] = 0
            self._cached_mask = mask
        return self._cached_mask
    
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Apply action-only constraint to logits.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len) - unused but required
            scores: Logits for next token (batch_size, vocab_size)
            
        Returns:
            Modified scores with non-action tokens masked to -inf
        """
        mask = self._get_mask(scores)
        
        # Handle potential vocab size mismatch (e.g., model has fewer tokens)
        actual_vocab = scores.shape[-1]
        if actual_vocab != self.vocab_size:
            # Recompute mask for actual vocab size
            mask = torch.full(
                (actual_vocab,), 
                float('-inf'), 
                device=scores.device, 
                dtype=scores.dtype
            )
            action_start = max(0, actual_vocab - self.action_bins)
            mask[action_start:actual_vocab] = 0
        
        # Add mask to scores (broadcasting across batch dimension)
        return scores + mask
    
    def get_action_token_range(self) -> tuple[int, int]:
        """Return the (start, end) indices of valid action tokens."""
        return (self.action_start_idx, self.action_end_idx)
    
    def is_action_token(self, token_id: int) -> bool:
        """Check if a token ID is a valid action token."""
        return self.action_start_idx <= token_id < self.action_end_idx


class BatchActionLogitsProcessor:
    """
    Batch-aware version for vLLM or TensorRT-LLM integration.
    
    Optimized for high-throughput inference where multiple sequences
    are processed in parallel.
    """
    
    def __init__(self, vocab_size: int, action_bins: int = 256):
        self.processor = ActionOnlyLogitsProcessor(vocab_size, action_bins)
    
    def __call__(
        self,
        token_ids: list[list[int]],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Process batch of logits for vLLM compatibility.
        
        Args:
            token_ids: List of token ID sequences (one per batch item)
            logits: Batch logits tensor (batch_size, vocab_size)
            
        Returns:
            Constrained logits
        """
        # Convert to tensor format expected by base processor
        dummy_input_ids = torch.zeros(
            (len(token_ids), 1), 
            dtype=torch.long, 
            device=logits.device
        )
        return self.processor(dummy_input_ids, logits)


def create_processor_for_model(model_name_or_path: str) -> ActionOnlyLogitsProcessor:
    """
    Factory function to create a properly configured processor.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        
    Returns:
        Configured ActionOnlyLogitsProcessor
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    vocab_size = config.vocab_size
    
    return ActionOnlyLogitsProcessor(vocab_size=vocab_size, action_bins=256)
