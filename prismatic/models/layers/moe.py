import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class ActionMoELayer(nn.Module):
    """
    Mixture of Experts layer for action token prediction
    
    This design maintains compatibility with the token-based action prediction
    by routing the final token prediction through a mixture of experts.
    """
    def __init__(
        self, 
        hidden_size: int,
        vocab_size: int,
        num_experts: int = 4,
        num_selected_experts: int = 2,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
        # Router network (determines which experts to use)
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert networks (each produces logits for the full vocabulary)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(hidden_size * 4, vocab_size)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MoE
        
        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            output: MoE output of shape (batch_size, sequence_length, vocab_size)
            aux_loss: Dictionary containing auxiliary losses
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Reshape for routing
        flat_hidden = hidden_states.view(-1, self.hidden_size)
        
        # Get routing probabilities
        router_logits = self.router(flat_hidden)  # (batch_size*seq_len, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Calculate auxiliary load balancing loss
        # Encourage uniform expert utilization
        aux_loss = {
            "load_balancing_loss": 
                self._calculate_load_balancing_loss(routing_weights)
        }
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_selected_experts, dim=-1
        )
        
        # Normalize the selected expert weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs and blend them
        blended_output = torch.zeros(
            (batch_size * seq_len, self.vocab_size), 
            device=hidden_states.device
        )
        
        for expert_idx, expert in enumerate(self.experts):
            # Find which inputs use this expert
            # Create a boolean mask across all batch*seq elements
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Only run the expert for relevant inputs
                masked_hidden = flat_hidden[expert_mask]
                expert_output = expert(masked_hidden)  # (num_selected, vocab_size)
                
                # For each element using this expert, find its weight
                for i, is_selected in enumerate(expert_mask):
                    if is_selected:
                        # Find position of this expert in the top_k for this element
                        expert_pos = (top_k_indices[i] == expert_idx).nonzero(as_tuple=True)[0].item()
                        # Add weighted expert output
                        blended_output[i] += expert_output[expert_mask.nonzero(as_tuple=True)[0].tolist().index(i)] * top_k_weights[i, expert_pos]
        
        # Reshape back to original dimensions
        output = blended_output.view(batch_size, seq_len, self.vocab_size)
        
        return output, aux_loss
    
    def _calculate_load_balancing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Calculate a load balancing loss to prevent expert collapse"""
        # Compute the fraction of tokens routed to each expert
        routing_prob = routing_weights.mean(dim=0)
        # Compute the fraction of routing weights assigned to each expert
        expert_usage = (routing_weights > 0).float().mean(dim=0)
        # Ideal balance would be uniform usage across experts
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        # L2 distance from the ideal uniform distribution
        return ((expert_usage - target_usage) ** 2).mean()
        
    def from_pretrained_layers(self, lm_head):
        """Initialize experts from an existing language model head"""
        # Clone and slightly perturb the weights to create diverse experts
        for i, expert in enumerate(self.experts):
            # Get the final layer that produces the logits
            with torch.no_grad():
                # For the first layer
                if hasattr(expert[0], 'weight'):
                    if i == 0:
                        # First expert gets exact original weights
                        expert[0].weight.data = lm_head.weight.data[:self.hidden_size * 4, :].clone()
                        if hasattr(expert[0], 'bias') and hasattr(lm_head, 'bias'):
                            expert[0].bias.data = lm_head.bias.data[:self.hidden_size * 4].clone()
                    else:
                        # Other experts get slightly perturbed weights
                        expert[0].weight.data = lm_head.weight.data[:self.hidden_size * 4, :].clone()
                        expert[0].weight.data += (0.02 * i) * torch.randn_like(expert[0].weight.data) * expert[0].weight.data.std()
                        if hasattr(expert[0], 'bias') and hasattr(lm_head, 'bias'):
                            expert[0].bias.data = lm_head.bias.data[:self.hidden_size * 4].clone()
                
                # For the final layer (to vocab size)
                if hasattr(expert[-1], 'weight'):
                    if i == 0:
                        # First expert gets exact original weights
                        expert[-1].weight.data = lm_head.weight.data.clone()
                        if hasattr(expert[-1], 'bias') and hasattr(lm_head, 'bias'):
                            expert[-1].bias.data = lm_head.bias.data.clone()
                    else:
                        # Other experts get more significant perturbations for diversity
                        expert[-1].weight.data = lm_head.weight.data.clone()
                        expert[-1].weight.data += (0.05 * i) * torch.randn_like(expert[-1].weight.data) * expert[-1].weight.data.std()
                        if hasattr(expert[-1], 'bias') and hasattr(lm_head, 'bias'):
                            expert[-1].bias.data = lm_head.bias.data.clone()
        
        return self 