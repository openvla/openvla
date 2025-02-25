import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        num_selected_experts: int = 2,
        expert_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
        # Gate network to select experts
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 4 * input_dim),
                nn.GELU(),
                nn.Linear(4 * input_dim, output_dim)
            ) for _ in range(num_experts)
        ])

        # Initialize from pretrained weights if provided
        if expert_weights is not None:
            self._initialize_from_pretrained(expert_weights)

    def _initialize_from_pretrained(self, weights):
        # Clone pretrained weights to all experts
        for expert in self.experts:
            for i, layer in enumerate(expert):
                if isinstance(layer, nn.Linear):
                    layer.weight.data = weights[i].weight.data.clone()
                    layer.bias.data = weights[i].bias.data.clone()
    
    def forward(self, x):
        # Calculate expert weights for each input
        gate_logits = self.gate(x)
        weights = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            weights, self.num_selected_experts, dim=-1
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute weighted sum of expert outputs
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Create mask for current expert
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                # Only compute for samples that use this expert
                expert_output = expert(x[mask])
                expert_weight = weights[mask, i].unsqueeze(-1)
                expert_outputs[mask] += expert_output * expert_weight
                
        return expert_outputs