from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, fan_in, fan_out, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.lora_A = nn.Parameter(torch.zeros((rank, fan_in)))
        self.lora_B = nn.Parameter(torch.zeros((fan_out, rank)))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but should not affect performance
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, X):             
        result = (self.lora_dropout(X) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        # result += F.linear(x, T(self.weight), bias=self.bias)
        return result

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )


class LoRA_MOE_LM(nn.Module): # for llm

    def __init__(
            self,
            args,
            lora_rank: int,
            lora_alpha: int,
            num_experts: int,
            original_module: nn.Module = None,
    ):
        super().__init__()
        self.args = args
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts

        d_model = original_module.gate_proj.in_features
        mlp_width = original_module.gate_proj.out_features
        self.original_module = original_module

        self.moe_gate = nn.ModuleList()
        self.moe_down = nn.ModuleList()
        self.moe_up = nn.ModuleList()
        self.original_module.gate_proj.weight.requires_grad = False
        self.original_module.down_proj.weight.requires_grad = False
        self.original_module.up_proj.weight.requires_grad = False

        for _ in range(num_experts):
            self.moe_gate.append(LoRALayer.from_linear(
                nn.Linear(d_model, mlp_width),
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
            self.moe_up.append(LoRALayer.from_linear(
                nn.Linear(d_model, mlp_width),
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
            self.moe_down.append(LoRALayer.from_linear(
                nn.Linear(mlp_width, d_model),
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
        
        self.router = nn.Linear(d_model, self.num_experts)

    def forward_lora_moe(self, x, original_proj, routing, moe):
        original_out = original_proj(x)
        lora_out_per_expert = []
        for i in range(self.num_experts):
            lora_out_per_expert.append(moe[i](x))

        lora_out = torch.stack(lora_out_per_expert, 2)

        lora_out = (lora_out * routing[:,:,:,None]).sum(2)

        moe_out = original_out + lora_out
        return moe_out

    def forward_lora_moe_sparse(self, x, original_proj, routing_idx, moe):
        original_out = original_proj(x)

        lora_out = torch.zeros_like(original_out)
        for i in range(self.num_experts):
            id1, id2, _ = torch.where(routing_idx==i)
            lora_out[id1, id2] = moe[i](x[id1, id2])

        moe_out = original_out + lora_out
        return moe_out


    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        logits = self.router(x)
        routing = F.softmax(logits, dim=-1)
        index = routing.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        expert_choice = y_hard - routing.detach() + routing

        if self.args.dense_moe:
            gate_moe_out = self.forward_lora_moe(x, self.original_module.gate_proj, routing, self.moe_gate)
            up_moe_out = self.forward_lora_moe(x, self.original_module.up_proj, routing, self.moe_up)
        else:
            gate_moe_out = self.forward_lora_moe_sparse(x, self.original_module.gate_proj, index, self.moe_gate)
            up_moe_out = self.forward_lora_moe_sparse(x, self.original_module.up_proj, index, self.moe_up)

        x = self.original_module.act_fn(gate_moe_out) * up_moe_out
        
        if self.args.dense_moe:
            x = self.forward_lora_moe(x, self.original_module.down_proj, routing, self.moe_down)
        else:
            x = self.forward_lora_moe_sparse(x, self.original_module.down_proj, index, self.moe_down)
        return x, (routing, expert_choice)