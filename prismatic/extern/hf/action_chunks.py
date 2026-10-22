"""Optional continuous action readout, independent of QuIC parameterization."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.utils import ModelOutput


def validate_action_head(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"type", "horizon", "hidden_dim"}
            or value["type"] != "mlp_chunk"
            or type(value["horizon"]) is not int or not 1 <= value["horizon"] <= 256
            or type(value["hidden_dim"]) is not int or not 1 <= value["hidden_dim"] <= 16384):
        raise ValueError("action_head requires type=mlp_chunk, horizon=1..256, hidden_dim=1..16384")
    return dict(value)


class ActionChunkHead(nn.Module):
    """Read the final prompt state and predict H consecutive seven dimensional actions."""

    def __init__(self, input_dim, specification):
        super().__init__()
        self.specification = validate_action_head(specification)
        self.input = nn.Linear(input_dim, specification["hidden_dim"])
        self.activation = nn.GELU()
        self.output = nn.Linear(specification["hidden_dim"], 7 * specification["horizon"])

    def forward(self, state):
        # Keep the small trainable readout in FP32 in the reference BF16 host.
        with torch.autocast(device_type=state.device.type, enabled=False):
            state = state.to(dtype=self.input.weight.dtype)
            return self.output(self.activation(self.input(state))).reshape(state.shape[0], self.specification["horizon"], 7)


@dataclass
class ActionChunkOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None


def forward_action_chunk(model, input_ids, pixel_values, attention_mask=None, actions=None, action_mask=None):
    """Use the decoder directly: no vocabulary logits, KV cache or ground truth action inputs."""
    if input_ids is None or pixel_values is None or input_ids.ndim != 2:
        raise ValueError("Action chunks require batched prompt input_ids and pixel_values")
    if pixel_values.shape[0] != input_ids.shape[0] or input_ids.shape[1] < 2:
        raise ValueError("Action chunks need matching image/prompt batches and at least two prompt tokens")
    if input_ids.shape[1] > model.config.llm_max_length:
        raise ValueError("Action prompt exceeds configured context")
    if attention_mask is None:
        attention_mask = input_ids.ne(model.config.pad_token_id)
    if attention_mask.shape != input_ids.shape or not attention_mask.any(-1).all():
        raise ValueError("Each action prompt needs a nonempty matching attention mask")
    # Prismatic inserts visual tokens immediately after BOS. Training and inference
    # share this layout and select the last *valid* prompt token, including padding.
    patches = model.projector(model.vision_backbone(pixel_values))
    embeddings = model.get_input_embeddings()(input_ids)
    multimodal = torch.cat((embeddings[:, :1], patches, embeddings[:, 1:]), dim=1)
    visual_mask = attention_mask.new_ones((input_ids.shape[0], patches.shape[1]))
    mask = torch.cat((attention_mask[:, :1], visual_mask, attention_mask[:, 1:]), dim=1)
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(~mask.bool(), 0)
    output = model.language_model.model(inputs_embeds=multimodal, attention_mask=mask,
        position_ids=positions, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    last = torch.arange(mask.shape[1], device=mask.device).expand_as(mask).masked_fill(~mask.bool(), -1).max(-1).values
    state = output.last_hidden_state[torch.arange(input_ids.shape[0], device=mask.device), last]
    predicted = model.action_head(state)
    loss = None
    if actions is not None:
        if actions.shape != predicted.shape or not torch.isfinite(actions).all():
            raise ValueError("Chunk targets must be finite [B,H,7] actions")
        if (action_mask is None or action_mask.dtype != torch.bool
                or action_mask.shape != predicted.shape[:2] or not action_mask.any(-1).all()):
            raise ValueError("Each chunk needs a nonempty boolean [B,H] validity mask")
        # Accumulate regression in FP32; padded future steps contribute no loss.
        errors = (predicted.float() - actions.float()).abs()
        loss = errors.masked_select(action_mask.unsqueeze(-1).expand_as(errors)).mean()
    elif action_mask is not None:
        raise ValueError("action_mask requires action targets")
    return ActionChunkOutput(loss=loss, actions=predicted)
