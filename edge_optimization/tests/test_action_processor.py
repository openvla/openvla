"""
Tests for ActionOnlyLogitsProcessor

These tests validate the action-only constraint works correctly
WITHOUT requiring the full model to be loaded (fast unit tests).
"""

import pytest
import numpy as np

# Mock torch if not available (for CPU-only testing)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


class TestActionOnlyLogitsProcessor:
    """Test suite for ActionOnlyLogitsProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a standard processor instance."""
        from edge_optimization.action_logits_processor import ActionOnlyLogitsProcessor
        return ActionOnlyLogitsProcessor(vocab_size=32000, action_bins=256)
    
    @pytest.fixture
    def sample_logits(self):
        """Create sample logits tensor."""
        # Batch of 2, vocab size of 32000
        return torch.randn(2, 32000)
    
    def test_processor_initialization(self, processor):
        """Test processor initializes with correct parameters."""
        assert processor.vocab_size == 32000
        assert processor.action_bins == 256
        assert processor.action_start_idx == 32000 - 256
        assert processor.action_end_idx == 32000
    
    def test_action_token_range(self, processor):
        """Test action token range calculation."""
        start, end = processor.get_action_token_range()
        assert start == 31744  # 32000 - 256
        assert end == 32000
    
    def test_is_action_token(self, processor):
        """Test action token identification."""
        # Non-action tokens
        assert not processor.is_action_token(0)
        assert not processor.is_action_token(1000)
        assert not processor.is_action_token(31743)
        
        # Action tokens
        assert processor.is_action_token(31744)
        assert processor.is_action_token(31900)
        assert processor.is_action_token(31999)
        
        # Boundary
        assert not processor.is_action_token(32000)
    
    def test_mask_non_action_tokens(self, processor, sample_logits):
        """Test that non-action tokens are masked to -inf."""
        input_ids = torch.zeros(2, 10, dtype=torch.long)  # Dummy input
        
        masked = processor(input_ids, sample_logits)
        
        # Non-action tokens should be -inf
        assert torch.all(masked[:, :31744] == float('-inf'))
        
        # Action tokens should be unchanged
        original_action_logits = sample_logits[:, 31744:]
        masked_action_logits = masked[:, 31744:]
        assert torch.allclose(original_action_logits, masked_action_logits)
    
    def test_softmax_only_action_tokens(self, processor, sample_logits):
        """Test that softmax gives 0 probability to non-action tokens."""
        input_ids = torch.zeros(2, 10, dtype=torch.long)
        
        masked = processor(input_ids, sample_logits)
        probs = torch.softmax(masked, dim=-1)
        
        # Non-action tokens should have 0 probability
        assert torch.all(probs[:, :31744] == 0)
        
        # Action tokens should have positive probability
        assert torch.all(probs[:, 31744:] > 0)
        
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    
    def test_batch_processing(self, processor):
        """Test processing works with different batch sizes."""
        for batch_size in [1, 4, 16]:
            logits = torch.randn(batch_size, 32000)
            input_ids = torch.zeros(batch_size, 5, dtype=torch.long)
            
            masked = processor(input_ids, logits)
            
            assert masked.shape == (batch_size, 32000)
            assert torch.all(masked[:, :31744] == float('-inf'))
    
    def test_device_handling(self, processor):
        """Test processor works on different devices."""
        logits_cpu = torch.randn(2, 32000)
        input_ids = torch.zeros(2, 5, dtype=torch.long)
        
        masked_cpu = processor(input_ids, logits_cpu)
        assert masked_cpu.device == logits_cpu.device
        
        if torch.cuda.is_available():
            logits_gpu = logits_cpu.cuda()
            input_ids_gpu = input_ids.cuda()
            
            masked_gpu = processor(input_ids_gpu, logits_gpu)
            assert masked_gpu.device == logits_gpu.device
    
    def test_different_vocab_sizes(self):
        """Test processor adapts to different vocab sizes."""
        from edge_optimization.action_logits_processor import ActionOnlyLogitsProcessor
        
        for vocab_size in [32000, 32256, 50000]:
            processor = ActionOnlyLogitsProcessor(vocab_size=vocab_size)
            logits = torch.randn(1, vocab_size)
            input_ids = torch.zeros(1, 5, dtype=torch.long)
            
            masked = processor(input_ids, logits)
            
            # Check correct number of unmasked tokens
            unmasked_count = (masked[0] != float('-inf')).sum()
            assert unmasked_count == 256
            
            # Check they're the last 256
            assert torch.all(masked[0, -256:] != float('-inf'))


class TestBatchActionLogitsProcessor:
    """Test suite for batch-aware processor."""
    
    def test_vllm_compatibility(self):
        """Test vLLM-style interface works."""
        from edge_optimization.action_logits_processor import BatchActionLogitsProcessor
        
        processor = BatchActionLogitsProcessor(vocab_size=32000)
        
        # vLLM-style input
        token_ids = [[1, 2, 3], [4, 5, 6]]
        logits = torch.randn(2, 32000)
        
        masked = processor(token_ids, logits)
        
        assert masked.shape == (2, 32000)
        assert torch.all(masked[:, :31744] == float('-inf'))


class TestCreateProcessorForModel:
    """Test factory function."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_create_from_hub(self):
        """Test creating processor from HuggingFace model."""
        from edge_optimization.action_logits_processor import create_processor_for_model
        
        processor = create_processor_for_model("openvla/openvla-7b")
        assert processor.vocab_size > 0
        assert processor.action_bins == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
