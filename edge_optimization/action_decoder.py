"""
Action Decoder for OpenVLA

Converts discrete action tokens back to continuous robot joint values.
Integrates with OpenVLA's ActionTokenizer for compatibility.

Usage:
    from edge_optimization import ActionDecoder
    
    decoder = ActionDecoder(vocab_size=32000)
    actions = decoder.decode(token_ids)  # Returns numpy array of 7-DOF actions
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class ActionStats:
    """Dataset-specific normalization statistics for actions."""
    mean: np.ndarray
    std: np.ndarray
    min_val: float = -1.0
    max_val: float = 1.0


# Pre-defined stats for common datasets (from OpenVLA training)
DATASET_STATS = {
    "bridge_orig": ActionStats(
        mean=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        std=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ),
    # Add more dataset stats as needed
}


class ActionDecoder:
    """
    Decodes OpenVLA action tokens to continuous robot actions.
    
    OpenVLA uses 256 uniform bins to discretize continuous actions in [-1, 1].
    The action tokens are mapped to the LAST 256 tokens of the vocabulary.
    
    Process:
        1. Token ID → Bin index: bin = vocab_size - token_id
        2. Bin index → Bin center: action = bin_centers[bin - 1]
        3. Apply unnormalization using dataset stats
    
    Args:
        vocab_size: Model vocabulary size (e.g., 32000)
        n_bins: Number of discretization bins (default: 256)
        min_action: Minimum action value before discretization (default: -1)
        max_action: Maximum action value before discretization (default: 1)
        action_dim: Number of action dimensions (default: 7 for 6-DOF + gripper)
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
        action_dim: int = 7
    ):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action
        self.action_dim = action_dim
        
        # Pre-compute bin edges and centers (matches OpenVLA ActionTokenizer)
        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # Action token range
        self.action_token_start = vocab_size - n_bins

    
    def decode_token_ids(
        self, 
        token_ids: Union[np.ndarray, list],
        unnorm_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Decode token IDs to continuous action values.
        
        Args:
            token_ids: Array of token IDs with shape (action_dim,) or (batch, action_dim)
            unnorm_key: Optional dataset key for unnormalization
            
        Returns:
            Continuous actions with same batch shape
        """
        token_ids = np.asarray(token_ids)
        original_shape = token_ids.shape
        
        # Flatten for processing
        flat_ids = token_ids.flatten()
        
        # Convert token IDs to bin indices
        # OpenVLA formula: discretized = vocab_size - token_id
        discretized = self.vocab_size - flat_ids
        
        # Clip to valid range and map to bin centers
        # Subtract 1 because digitize returns 1-indexed bins
        bin_indices = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        actions = self.bin_centers[bin_indices]
        
        # Reshape back
        actions = actions.reshape(original_shape)
        
        # Apply unnormalization if requested
        if unnorm_key is not None:
            actions = self.unnormalize(actions, unnorm_key)
        
        return actions
    
    def unnormalize(
        self, 
        actions: np.ndarray, 
        dataset_key: str
    ) -> np.ndarray:
        """
        Apply dataset-specific unnormalization.
        
        Args:
            actions: Normalized actions in [-1, 1]
            dataset_key: Key for dataset statistics
            
        Returns:
            Unnormalized actions in original scale
        """
        if dataset_key not in DATASET_STATS:
            # Return as-is if no stats available
            return actions
        
        stats = DATASET_STATS[dataset_key]
        return actions * stats.std + stats.mean
    
    def decode_sequence(
        self,
        token_sequence: Union[np.ndarray, list],
        unnorm_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Decode a full sequence of action tokens.
        
        Assumes tokens come in groups of action_dim consecutive tokens.
        
        Args:
            token_sequence: Flat sequence of token IDs
            unnorm_key: Optional dataset key for unnormalization
            
        Returns:
            Actions with shape (n_timesteps, action_dim)
        """
        token_sequence = np.asarray(token_sequence)
        
        if len(token_sequence) % self.action_dim != 0:
            raise ValueError(
                f"Token sequence length {len(token_sequence)} is not divisible "
                f"by action_dim {self.action_dim}"
            )
        
        n_timesteps = len(token_sequence) // self.action_dim
        reshaped = token_sequence.reshape(n_timesteps, self.action_dim)
        
        return self.decode_token_ids(reshaped, unnorm_key)
    
    def is_valid_action_token(self, token_id: int) -> bool:
        """Check if a token ID is within the action token range."""
        return self.action_token_start <= token_id < self.vocab_size


def create_decoder_for_model(model_name_or_path: str) -> ActionDecoder:
    """
    Factory function to create a properly configured decoder.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        
    Returns:
        Configured ActionDecoder
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    vocab_size = config.vocab_size
    
    return ActionDecoder(vocab_size=vocab_size)
