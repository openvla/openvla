from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prismatic.models.vlas.openvla import OpenVLA
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer


class MoEActionLayer(nn.Module):
    """MoE layer specific for action prediction"""
    def __init__(
        self, 
        hidden_size: int,
        action_dim: int,
        num_experts: int = 4,
        num_selected_experts: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # Task/context router - determines which experts to use based on context
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert MLPs - each one specializes in different action patterns
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, action_dim * self.bin_centers.shape[0])
            ) for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(
            routing_probs, self.num_selected_experts, dim=-1
        )
        
        # Normalize the probabilities of selected experts
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Calculate weighted sum of expert outputs
        outputs = torch.zeros(
            (batch_size, self.action_dim, self.bin_centers.shape[0]), 
            device=hidden_states.device
        )
        
        for expert_idx, expert in enumerate(self.experts):
            # Find which batch samples use this expert
            batch_indices = (topk_indices == expert_idx).any(dim=-1).nonzero().squeeze(-1)
            
            if len(batch_indices) > 0:
                # Only run expert for relevant samples
                expert_output = expert(hidden_states[batch_indices])
                expert_output = expert_output.view(-1, self.action_dim, self.bin_centers.shape[0])
                
                # Get weights for this expert
                expert_probs = torch.zeros(batch_size, device=hidden_states.device)
                for i, idx in enumerate(batch_indices):
                    mask = (topk_indices[idx] == expert_idx).nonzero().item()
                    expert_probs[idx] = topk_probs[idx, mask]
                
                # Add weighted expert output
                for i, idx in enumerate(batch_indices):
                    outputs[idx] += expert_output[i] * expert_probs[idx]
        
        return outputs


class MoEOpenVLAForActionPrediction(OpenVLAForActionPrediction):
    """OpenVLA with Mixture of Experts for action prediction"""
    
    def __init__(
        self, 
        config, 
        num_experts: int = 4,
        num_selected_experts: int = 2,
        task_conditional: bool = True,
    ):
        super().__init__(config)
        
        # Create MoE action head
        self.action_moe = MoEActionLayer(
            hidden_size=self.config.text_config.hidden_size,
            action_dim=self.get_action_dim(),  # Max action dimension
            num_experts=num_experts,
            num_selected_experts=num_selected_experts
        )
        
        # Create task embedding if using task conditioning
        self.task_conditional = task_conditional
        if task_conditional:
            self.task_embeddings = nn.Embedding(
                len(self.norm_stats),  # Number of tasks
                self.config.text_config.hidden_size
            )
            # Create task mapping
            self.task_id_map = {task: idx for idx, task in enumerate(self.norm_stats.keys())}

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get maximum action dimension from all tasks if unnorm_key is None"""
        if unnorm_key is None:
            return max(len(stats["mean"]) for task, stats in self.norm_stats.items() 
                      for action_type, stats in self.norm_stats[task].items())
        return len(self.get_action_stats(unnorm_key)["mean"])
            
    def predict_action(
        self, 
        input_ids: Optional[torch.LongTensor] = None, 
        unnorm_key: Optional[str] = None, 
        **kwargs
    ) -> np.ndarray:
        """Enhanced action prediction using MoE"""
        # Get last hidden state from model
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        # Incorporate task information if using task conditioning
        if self.task_conditional and unnorm_key is not None:
            task_id = self.task_id_map.get(unnorm_key, 0)
            task_embedding = self.task_embeddings(torch.tensor([task_id], device=last_hidden_state.device))
            # Combine with hidden state (simple addition, but could use more complex fusion)
            last_hidden_state = last_hidden_state + task_embedding
        
        # Pass through MoE action layer
        action_logits = self.action_moe(last_hidden_state)
        
        # Get highest probability bin for each action dimension
        action_bin_indices = torch.argmax(action_logits, dim=-1)
        
        # Convert to normalized actions using bin centers
        normalized_actions = torch.tensor(
            [self.bin_centers[idx.item()] for idx in action_bin_indices],
            device=action_bin_indices.device
        )
        
        # Unnormalize actions if unnorm_key provided
        if unnorm_key is not None:
            action_norm_stats = self.get_action_stats(unnorm_key)
            action_mean = torch.tensor(action_norm_stats["mean"], device=normalized_actions.device)
            action_scale = torch.tensor(action_norm_stats["scale"], device=normalized_actions.device)
            return (normalized_actions * action_scale + action_mean).cpu().numpy()
        
        return normalized_actions.cpu().numpy()

    def from_pretrained_model(self, pretrained_model):
        """Initialize from pretrained model"""
        # Copy all parameters except the new MoE layers
        self.load_state_dict(pretrained_model.state_dict(), strict=False)
        
        # Now let's initialize the MoE experts with existing knowledge
        # Get the final layer from the language model that produces logits
        if hasattr(pretrained_model, 'lm_head'):
            # Initialize all experts with variations of the original weights
            lm_head = pretrained_model.lm_head
            for i, expert in enumerate(self.action_moe.experts):
                # Slightly different initialization for each expert
                with torch.no_grad():
                    # First layer
                    expert[0].weight.data = lm_head.weight.data[:self.hidden_size * 2, :].clone()
                    expert[0].weight.data += 0.01 * (i+1) * torch.randn_like(expert[0].weight.data)
                    expert[0].bias.data.zero_()
                    
                    # Last layer initialized with small random weights
                    expert[2].weight.data.normal_(mean=0.0, std=0.02)
                    expert[2].bias.data.zero_()
        
        return self