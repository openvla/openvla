from typing import Dict, Optional
import torch
import torch.nn as nn

from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.moe.moe import MoELayer

class MoEOpenVLA(OpenVLA):
    def __init__(
        self,
        *args,
        num_experts: int = 4,
        num_selected_experts: int = 2,
        moe_in_transformer: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Extract original MLP weights for initialization
        original_mlp = self.projector
        
        # Replace projector with MoE layer
        self.projector = MoELayer(
            input_dim=self.vision_backbone.embed_dim,
            output_dim=self.llm_backbone.embed_dim,
            num_experts=num_experts,
            num_selected_experts=num_selected_experts,
            expert_weights=original_mlp
        )
        
        # Optionally add MoE to transformer blocks
        if moe_in_transformer:
            self._convert_transformer_to_moe(
                num_experts=num_experts,
                num_selected_experts=num_selected_experts
            )

    def _convert_transformer_to_moe(self, num_experts: int, num_selected_experts: int):
        """Convert transformer MLP blocks to MoE layers"""
        for layer in self.llm_backbone.transformer.layers:
            # Store original MLP weights
            original_mlp = layer.mlp
            
            # Replace with MoE
            layer.mlp = MoELayer(
                input_dim=self.llm_backbone.embed_dim,
                output_dim=self.llm_backbone.embed_dim,
                num_experts=num_experts,
                num_selected_experts=num_selected_experts,
                expert_weights=original_mlp
            )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # First load the original model
        model = super().from_pretrained(*args, **kwargs)
        
        # Convert to MoE version while preserving weights
        moe_model = cls(
            *args,
            num_experts=kwargs.get('num_experts', 4),
            num_selected_experts=kwargs.get('num_selected_experts', 2),
            moe_in_transformer=kwargs.get('moe_in_transformer', False),
            **kwargs
        )
        
        return moe_model