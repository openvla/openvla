"""
Tests for ActionDecoder

Tests token-to-continuous action conversion without requiring model loading.
"""

import pytest
import numpy as np


class TestActionDecoder:
    """Test suite for ActionDecoder."""
    
    @pytest.fixture
    def decoder(self):
        """Create a standard decoder instance."""
        from edge_optimization.action_decoder import ActionDecoder
        return ActionDecoder(vocab_size=32000, n_bins=256)
    
    def test_decoder_initialization(self, decoder):
        """Test decoder initializes with correct parameters."""
        assert decoder.vocab_size == 32000
        assert decoder.n_bins == 256
        assert decoder.action_dim == 7
        assert len(decoder.bin_centers) == 255  # n_bins - 1 centers
    
    def test_bin_centers_range(self, decoder):
        """Test bin centers are in expected range."""
        assert decoder.bin_centers[0] >= -1.0
        assert decoder.bin_centers[-1] <= 1.0
        
        # Check centers are evenly spaced
        diffs = np.diff(decoder.bin_centers)
        assert np.allclose(diffs, diffs[0])
    
    def test_decode_single_action(self, decoder):
        """Test decoding a single 7-DOF action."""
        # Token IDs near the middle of action range
        # vocab_size - token_id = bin_index
        # For vocab_size=32000, action tokens are 31744-31999
        # Token 31872 -> bin 128 (middle) -> action ~0
        middle_token = 32000 - 128
        token_ids = np.array([middle_token] * 7)
        
        actions = decoder.decode_token_ids(token_ids)
        
        assert actions.shape == (7,)
        assert np.all(np.abs(actions) <= 1.0)
        # Middle bin should give action near 0
        assert np.allclose(actions, 0, atol=0.01)
    
    def test_decode_extreme_values(self, decoder):
        """Test decoding tokens at extremes of action range."""
        # High tokens (low bin indices) -> negative actions
        high_tokens = np.array([31999] * 7)  # bin = 1
        low_actions = decoder.decode_token_ids(high_tokens)
        assert np.all(low_actions < 0)
        
        # Low tokens (high bin indices) -> positive actions
        low_tokens = np.array([31745] * 7)  # bin = 255
        high_actions = decoder.decode_token_ids(low_tokens)
        assert np.all(high_actions > 0)
    
    def test_decode_batch(self, decoder):
        """Test decoding batch of actions."""
        # Batch of 3 timesteps, each with 7 actions
        batch_tokens = np.array([
            [31872] * 7,  # Middle actions
            [31745] * 7,  # High actions
            [31999] * 7,  # Low actions
        ])
        
        actions = decoder.decode_token_ids(batch_tokens)
        
        assert actions.shape == (3, 7)
        assert np.all(np.abs(actions) <= 1.0)
    
    def test_decode_sequence(self, decoder):
        """Test decoding flat token sequence."""
        # 2 timesteps * 7 actions = 14 tokens
        flat_tokens = [31872] * 14
        
        actions = decoder.decode_sequence(flat_tokens)
        
        assert actions.shape == (2, 7)
    
    def test_invalid_sequence_length(self, decoder):
        """Test error on invalid sequence length."""
        # 10 tokens is not divisible by 7
        bad_tokens = [31872] * 10
        
        with pytest.raises(ValueError, match="not divisible"):
            decoder.decode_sequence(bad_tokens)
    
    def test_is_valid_action_token(self, decoder):
        """Test action token validation."""
        # Valid action tokens
        assert decoder.is_valid_action_token(31744)
        assert decoder.is_valid_action_token(31872)
        assert decoder.is_valid_action_token(31999)
        
        # Invalid tokens
        assert not decoder.is_valid_action_token(0)
        assert not decoder.is_valid_action_token(31743)
        assert not decoder.is_valid_action_token(32000)
    
    def test_token_to_action_mapping_consistency(self, decoder):
        """Test that mapping is consistent and monotonic."""
        # Lower token IDs should give higher continuous values
        tokens = np.arange(31745, 31745 + 10)
        prev_action = float('inf')
        
        for t in tokens:
            action = decoder.decode_token_ids(np.array([t]))[0]
            assert action < prev_action  # Should decrease as token increases
            prev_action = action


class TestDatasetStats:
    """Test dataset statistics handling."""
    
    def test_stats_available(self):
        """Test that default stats are available."""
        from edge_optimization.action_decoder import DATASET_STATS
        
        assert "bridge_orig" in DATASET_STATS
    
    def test_unnormalize_identity(self):
        """Test unnormalization with identity stats."""
        from edge_optimization.action_decoder import ActionDecoder
        
        decoder = ActionDecoder(vocab_size=32000)
        actions = np.array([0.5, -0.3, 0.0, 0.1, -0.1, 0.2, -0.2])
        
        # bridge_orig has identity transform
        unnormed = decoder.unnormalize(actions, "bridge_orig")
        
        assert np.allclose(actions, unnormed)
    
    def test_unnormalize_unknown_dataset(self):
        """Test unnormalization with unknown dataset returns unchanged."""
        from edge_optimization.action_decoder import ActionDecoder
        
        decoder = ActionDecoder(vocab_size=32000)
        actions = np.array([0.5, -0.3, 0.0, 0.1, -0.1, 0.2, -0.2])
        
        unnormed = decoder.unnormalize(actions, "unknown_dataset")
        
        assert np.allclose(actions, unnormed)


class TestCreateDecoderForModel:
    """Test factory function."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_create_from_hub(self):
        """Test creating decoder from HuggingFace model."""
        from edge_optimization.action_decoder import create_decoder_for_model
        
        decoder = create_decoder_for_model("openvla/openvla-7b")
        assert decoder.vocab_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
