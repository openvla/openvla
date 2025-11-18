import json
import os
import time

import numpy as np
import tensorflow as tf
import torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

import matplotlib.pyplot as plt
from typing import Dict, List
import re

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

import re

def _focus_top_indices_from_sims(sims: torch.Tensor, pct: float) -> List[int]:
    pct = float(max(1e-6, min(1.0, pct)))
    N = sims.numel()
    k = max(1, int(round(pct * N)))
    return torch.topk(sims, k=k, largest=True, sorted=False).indices.detach().cpu().tolist()

def _set_distance_jaccard(curr_idx: List[int], prev_idx: List[int] = None) -> float:
    if not prev_idx:
        return 0.0
    A, B = set(curr_idx), set(prev_idx)
    union = len(A | B); inter = len(A & B)
    return 1.0 - (inter / float(max(1, union)))

def aggregate_visual_e_bar(
    out,
    t_anchor: torch.Tensor,
    tau: float = 0.3,
    use_attn_prior: bool = False,   
    prior_lambda: float = 0.3,  
    prior_entropy_gate: bool = True,
    eps: float = 1e-12,
):
    V = out.projector_features[0].float() 
    Vn = F.normalize(V, dim=-1)
    an = F.normalize(t_anchor, dim=-1)
    sims = torch.matmul(Vn, an)                  

    prior = None
    H_prior = None
    logits = sims / tau
   
    if use_attn_prior and hasattr(out, "cross_attentions") and out.cross_attentions:
        CA = out.cross_attentions[-1]             
        if CA.dim() == 4:
            if CA.size(0) == 1:                  
                A = CA[0].mean(dim=0)             
            else:                                  
                A = CA[:,0].mean(dim=0)          
            prior = A.mean(dim=0)                 
            prior = (prior / (prior.sum() + eps)).clamp_min(eps)
            if prior_entropy_gate:
                H_prior = float(-(prior * prior.log()).sum().item())
                logN = np.log(prior.numel())
                too_uniform = H_prior > (logN - 0.2)
                too_spiky  = H_prior < (logN - 2.0)
                if too_uniform or too_spiky:
                    prior_lambda = 0.0
            if prior_lambda > 0.0:
                logits = logits + prior_lambda * prior.log()

    w = torch.softmax(logits, dim=-1)   
    e_bar = torch.sum(V * w.unsqueeze(-1), dim=0) 
    return e_bar, w, sims, prior, H_prior

def update_episode_buffer(
    buf,
    e_bar_t,
    t_anchor,
    a_t,
    H_t,
    w_t,
    *,
    focus_bulk: Dict[str, Dict] = None,
):
    buf.setdefault("e_bar_seq", []).append(e_bar_t.detach().cpu().numpy())
    buf.setdefault("a_seq", []).append(a_t)
    buf.setdefault("H_seq", []).append(H_t)
    buf.setdefault("w_seq", []).append(w_t.detach().cpu().numpy())

    t_hat = F.normalize(t_anchor, dim=-1)
    par = float((e_bar_t * t_hat).sum().item())
    perp = float((e_bar_t - par * t_hat).norm().item())
    buf.setdefault("par_seq", []).append(par)
    buf.setdefault("perp_seq", []).append(perp)

    if focus_bulk:
        buf.setdefault("focus_idx_seq_map", {})
        buf.setdefault("focus_shift_pair_t_map", {})
        buf.setdefault("focus_shift_mean_map", {})
        pcts = list(buf.get("focus_top_pcts", []))

        for key, info in focus_bulk.items():
            pct = float(info["pct"])
            idx = list(map(int, info["idx"]))
            shift = float(info["shift"])

            if pct not in pcts:
                pcts.append(pct)
                pcts.sort()
            buf["focus_idx_seq_map"].setdefault(key, []).append(idx)

            seq = buf["focus_shift_pair_t_map"].setdefault(key, [])
            seq.append(shift)
            vals = [v for v in seq if v == v]
            buf["focus_shift_mean_map"][key] = float(np.mean(vals)) if vals else float("nan")

        buf["focus_top_pcts"] = pcts


def _normalize_text(s: str, lower: bool = True) -> str:
    if lower:
        s = s.lower()
    s = re.sub(r"\s+", " ", s.strip())
    return s

def find_min_coverage_tokens(
    tokenizer,
    prompt: str,
    phrase: str,
    *,
    add_special_tokens: bool = True,
    occurrence: int = 0,
    case_insensitive: bool = True,
) -> Dict[str, object]:

    text = prompt
    key  = phrase
    text_norm   = _normalize_text(text,  lower=case_insensitive)
    phrase_norm = _normalize_text(key,   lower=case_insensitive)

    start = -1; last = 0
    for _ in range(occurrence + 1):
        start = text_norm.find(phrase_norm, last)
        if start == -1:
            break
        last = start + 1
    end = -1 if start == -1 else start + len(phrase_norm)

    enc_full  = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors=None)
    input_ids = enc_full["input_ids"]
    if hasattr(input_ids, "tolist"): input_ids = input_ids.tolist()
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    plain_ids: List[int] = []
    plain_map: List[int] = []  
    for i, tid in enumerate(input_ids):
        if tid not in special_ids:
            plain_ids.append(tid)
            plain_map.append(i)

    def decode_norm(i: int, j: int) -> str:
        s = tokenizer.decode(plain_ids[i:j], skip_special_tokens=True)
        return _normalize_text(s, lower=case_insensitive)

    N = len(plain_ids)
    if N == 0:
        return {"indices": [], "weights": [], "span": (start, end),
                "input_ids": input_ids, "offset_mapping": None}

    best_i = best_j = -1
    for i in range(N):
        for j in range(i+1, min(N, i+256)+1):
            sub_norm = decode_norm(i, j)
            if phrase_norm and phrase_norm in sub_norm:
                ii, jj = i, j
                while ii+1 < jj:
                    sub2 = decode_norm(ii+1, jj)
                    if phrase_norm in sub2:
                        ii += 1
                    else:
                        break
                while ii < jj-1:
                    sub2 = decode_norm(ii, jj-1)
                    if phrase_norm in sub2:
                        jj -= 1
                    else:
                        break
                best_i, best_j = ii, jj
                break
        if best_i != -1:
            break

    def coverage_score(sub: str, ref: str) -> float:
        set_sub = set(sub)
        set_ref = set(ref)
        inter = len(set_sub & set_ref)
        cov   = inter / max(1, len(set_ref))
        iou   = inter / max(1, len(set_sub | set_ref))
        return 0.7 * cov + 0.3 * iou

    if best_i == -1:
        best_score, best_len = -1.0, 10**9
        for i in range(N):
            for j in range(i+1, min(N, i+256)+1):
                sub_norm = decode_norm(i, j)
                sc = coverage_score(sub_norm, phrase_norm)
                L  = j - i
                if (sc > best_score) or (sc == best_score and L < best_len):
                    best_score, best_len = sc, L
                    best_i, best_j = i, j

        if best_score < 0.5:
            return {"indices": [], "weights": [], "span": (start, end),
                    "input_ids": input_ids, "offset_mapping": None}

    hit_plain = list(range(best_i, best_j))
    hit = [plain_map[k] for k in hit_plain]


    weights: List[float] = []
    token_strs = [tokenizer.decode([plain_ids[k]], skip_special_tokens=True) for k in hit_plain]
    token_strs = [_normalize_text(s, lower=case_insensitive) for s in token_strs]

    set_ref = set(phrase_norm)
    for s in token_strs:
        inter = len(set(s) & set_ref)
        denom = max(1, len(s))
        weights.append(inter / denom)

    ssum = sum(weights)
    if ssum <= 1e-12:
        weights = [1.0/len(hit)] * len(hit)
    else:
        weights = [w / ssum for w in weights]

    return {
        "indices": hit,          
        "weights": weights, 
        "span": (start, end),   
        "input_ids": input_ids,  
        "offset_mapping": None,    
    }


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(cfg.device)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, task_id, eps, t, eps_buf, device, center_crop=False, *, focus_top_pcts: tuple = (0.01, 0.03, 0.05, 0.10), focus_top_pct: float = 0.10):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    
    # Process inputs.
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = vla(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
    
    last_h = out.hidden_states[-1][0].float()
    N_patches = out.projector_features[0].shape[0]

    res = find_min_coverage_tokens(processor.tokenizer, prompt, phrase=task_label.lower())

    idx = res["indices"]
    ws = res["weights"]
    idx_mm = [(j if j == 0 else j + N_patches) for j in idx]

    print("[anchor] tokens:", processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][idx]))
    
    # get anchor
    if len(idx) > 0:
        print('anchor successfully, using the key tokens.')
        ws_t = torch.tensor(ws, device=last_h.device, dtype=last_h.dtype).unsqueeze(-1)
        t_anchor = (last_h[idx_mm] * ws_t).sum(dim=0)                                     
    else:
        print('cannot anchor.., using all tokens.')
        t_anchor = last_h.mean(dim=0)
    
    e_bar, w_t, sims, prior, H_prior = aggregate_visual_e_bar(out, t_anchor)

    pct = float(max(1e-6, min(1.0, focus_top_pct)))
    curr_focus_idx = _focus_top_indices_from_sims(sims, pct=pct)
    prev_focus_idx = eps_buf["focus_idx_seq"][-1] if "focus_idx_seq" in eps_buf and len(eps_buf["focus_idx_seq"]) > 0 else None
    focus_shift_t = _set_distance_jaccard(curr_focus_idx, prev_focus_idx)
    
    a_t = float((w_t * sims).sum().item())
    H_t = float(-(w_t * (w_t.clamp_min(1e-12)).log()).sum().item())
    
    focus_bulk = {}
    for p in (0.01, 0.03, 0.05, 0.10):
        key = f"p{int(round(p*100))}"
        curr_idx = _focus_top_indices_from_sims(sims, pct=p)
        prev_idx = None
        if "focus_idx_seq_map" in eps_buf and key in eps_buf["focus_idx_seq_map"] and len(eps_buf["focus_idx_seq_map"][key]) > 0:
            prev_idx = eps_buf["focus_idx_seq_map"][key][-1]
        shift = _set_distance_jaccard(curr_idx, prev_idx)
        focus_bulk[key] = {"pct": p, "idx": curr_idx, "shift": shift}

    update_episode_buffer(
        eps_buf,
        e_bar,
        t_anchor,
        a_t,
        H_t,
        w_t,
        focus_bulk=focus_bulk,
    )
    
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    
    return action
