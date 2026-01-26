from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from .text_tower import TextCfg
from .vision_tower import VisionCfg, ResNetCfg
from .model import ClipCfg

def _to_array(t: Any) -> jnp.ndarray:
    """Best-effort convert weights (torch/np/jax) to jnp.ndarray."""
    if isinstance(t, jnp.ndarray):
        return t
    if isinstance(t, np.ndarray):
        return jnp.asarray(t)
    # Torch tensor compatibility without importing torch
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(t, attr):
            try:
                t = getattr(t, attr)()
            except Exception:
                pass
    return jnp.asarray(t)


def _get(sd: Dict[str, Any], key: str):
    if key in sd:
        return sd[key]
    key2 = key.replace("visual.", "")
    if key2 in sd:
        return sd[key2]
    raise KeyError(key)


def _split_qkv(x: Any):
    return jnp.split(_to_array(x), 3, axis=0)


def convert_text_state_dict(state_dict: Dict[str, Any], cfg: TextCfg) -> Dict:
    params = {}
    params["token_embedding"] = {"embedding": _to_array(state_dict["token_embedding.weight"])}
    params["positional_embedding"] = _to_array(state_dict["positional_embedding"])
    if cfg.cls_emb:
        params["cls_emb"] = _to_array(state_dict["cls_emb"])

    params["text_projection"] = _to_array(state_dict["text_projection"])

    blocks = {}
    for i in range(cfg.layers):
        pre = f"transformer.resblocks.{i}."
        blk = {}
        blk["ln1"] = {
            "scale": _to_array(state_dict[pre + "ln_1.weight"]),
            "bias": _to_array(state_dict[pre + "ln_1.bias"]),
        }
        blk["ln2"] = {
            "scale": _to_array(state_dict[pre + "ln_2.weight"]),
            "bias": _to_array(state_dict[pre + "ln_2.bias"]),
        }
        w_q, w_k, w_v = _split_qkv(state_dict[pre + "attn.in_proj_weight"])
        b_q, b_k, b_v = _split_qkv(state_dict[pre + "attn.in_proj_bias"])
        dim = cfg.width
        num_heads = cfg.heads
        head_dim = dim // num_heads

        def pack_qkv(w, b):
            k = _to_array(w).T.reshape(dim, num_heads, head_dim)
            bnp = _to_array(b).reshape(num_heads, head_dim)
            return {"kernel": k, "bias": bnp}

        attn = {
            "query": pack_qkv(w_q, b_q),
            "key": pack_qkv(w_k, b_k),
            "value": pack_qkv(w_v, b_v),
        }
        out_w = state_dict[pre + "attn.out_proj.weight"]  # [dim, dim]
        out_b = state_dict[pre + "attn.out_proj.bias"]
        out_kernel = _to_array(out_w).T.reshape(num_heads, head_dim, dim)  # [heads, head_dim, dim]
        attn["out"] = {
            "kernel": out_kernel,
            "bias": _to_array(out_b),
        }
        blk["attn"] = {"SelfAttention_0": attn}
        blk["mlp"] = {
            "fc1": {
                "kernel": _to_array(state_dict[pre + "mlp.c_fc.weight"]).T,
                "bias": _to_array(state_dict[pre + "mlp.c_fc.bias"]),
            },
            "fc2": {
                "kernel": _to_array(state_dict[pre + "mlp.c_proj.weight"]).T,
                "bias": _to_array(state_dict[pre + "mlp.c_proj.bias"]),
            },
        }
        blocks[f"resblocks_{i}"] = blk
    params.update(blocks)

    params["ln_final"] = {
        "scale": _to_array(state_dict["ln_final.weight"]),
        "bias": _to_array(state_dict["ln_final.bias"]),
    }
    return params


def convert_vit_state_dict(state_dict: Dict[str, Any], cfg: VisionCfg) -> Dict:
    params = {}
    # conv1: torch [out, in, kh, kw] -> Flax [kh, kw, in, out]
    conv_w = _get(state_dict, "visual.conv1.weight")
    params["conv1"] = {"kernel": _to_array(conv_w).transpose(2, 3, 1, 0)}

    params["positional_embedding"] = _to_array(_get(state_dict, "visual.positional_embedding"))
    if cfg.cls_token:
        params["cls_token"] = _to_array(_get(state_dict, "visual.class_embedding")).reshape(1, 1, -1)
    blocks = {}
    dim = cfg.width
    num_heads = cfg.heads
    head_dim = dim // num_heads
    for i in range(cfg.layers):
        pre = f"visual.transformer.resblocks.{i}."
        if pre + "ln_1.weight" not in state_dict:
            pre = f"transformer.resblocks.{i}."
        blk = {}
        blk["ln1"] = {
            "scale": _to_array(_get(state_dict, pre + "ln_1.weight")),
            "bias": _to_array(_get(state_dict, pre + "ln_1.bias")),
        }
        blk["ln2"] = {
            "scale": _to_array(_get(state_dict, pre + "ln_2.weight")),
            "bias": _to_array(_get(state_dict, pre + "ln_2.bias")),
        }
        w_q, w_k, w_v = _split_qkv(_get(state_dict, pre + "attn.in_proj_weight"))
        b_q, b_k, b_v = _split_qkv(_get(state_dict, pre + "attn.in_proj_bias"))

        def pack_qkv(w, b):
            k = _to_array(w).T.reshape(dim, num_heads, head_dim)
            bnp = _to_array(b).reshape(num_heads, head_dim)
            return {"kernel": k, "bias": bnp}

        attn = {
            "query": pack_qkv(w_q, b_q),
            "key": pack_qkv(w_k, b_k),
            "value": pack_qkv(w_v, b_v),
        }
        out_w = _get(state_dict, pre + "attn.out_proj.weight")
        out_b = _get(state_dict, pre + "attn.out_proj.bias")
        out_kernel = _to_array(out_w).T.reshape(num_heads, head_dim, dim)
        attn["out"] = {"kernel": out_kernel, "bias": _to_array(out_b)}
        blk["attn"] = {"SelfAttention_0": attn}
        blk["mlp"] = {
            "fc1": {
                "kernel": _to_array(_get(state_dict, pre + "mlp.c_fc.weight")).T,
                "bias": _to_array(_get(state_dict, pre + "mlp.c_fc.bias")),
            },
            "fc2": {
                "kernel": _to_array(_get(state_dict, pre + "mlp.c_proj.weight")).T,
                "bias": _to_array(_get(state_dict, pre + "mlp.c_proj.bias")),
            },
        }
        blocks[f"resblocks_{i}"] = blk
    params.update(blocks)
    params["ln_pre"] = {
        "scale": _to_array(_get(state_dict, "visual.ln_pre.weight")),
        "bias": _to_array(_get(state_dict, "visual.ln_pre.bias")),
    }
    params["ln_post"] = {
        "scale": _to_array(_get(state_dict, "visual.ln_post.weight")),
        "bias": _to_array(_get(state_dict, "visual.ln_post.bias")),
    }
    params["proj"] = _to_array(_get(state_dict, "visual.proj"))
    return params


def convert_resnet_state_dict(state_dict: Dict[str, Any], cfg: ResNetCfg) -> Tuple[Dict, Dict]:
    """
    Return (params, batch_stats) for Flax ModifiedResNet.
    """
    params = {}
    batch_stats = {}

    def g(key: str):
        if key in state_dict:
            return state_dict[key]
        pref = f"visual.{key}"
        if pref in state_dict:
            return state_dict[pref]
        raise KeyError(key)

    def map_bn(prefix):
        return {
            "scale": _to_array(g(prefix + "weight")),
            "bias": _to_array(g(prefix + "bias")),
        }, {
            "mean": _to_array(g(prefix + "running_mean")),
            "var": _to_array(g(prefix + "running_var")),
        }

    # stem
    params["conv1"] = {"kernel": _to_array(g("conv1.weight")).transpose(2, 3, 1, 0)}
    params["conv2"] = {"kernel": _to_array(g("conv2.weight")).transpose(2, 3, 1, 0)}
    params["conv3"] = {"kernel": _to_array(g("conv3.weight")).transpose(2, 3, 1, 0)}
    params["bn1"], batch_stats["bn1"] = map_bn("bn1.")
    params["bn2"], batch_stats["bn2"] = map_bn("bn2.")
    params["bn3"], batch_stats["bn3"] = map_bn("bn3.")

    # flattened blocks
    block_idx = 0
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        num_blocks = len([k for k in state_dict.keys() if k.startswith(f"{layer_name}.") and k.endswith("conv1.weight")])
        if num_blocks == 0:
            num_blocks = len([k for k in state_dict.keys() if k.startswith(f"visual.{layer_name}.") and k.endswith("conv1.weight")])
        for b in range(num_blocks):
            base = f"{layer_name}.{b}."
            block_params = {}
            block_stats = {}
            block_params["conv1"] = {"kernel": _to_array(g(base + "conv1.weight")).transpose(2, 3, 1, 0)}
            block_params["conv2"] = {"kernel": _to_array(g(base + "conv2.weight")).transpose(2, 3, 1, 0)}
            block_params["conv3"] = {"kernel": _to_array(g(base + "conv3.weight")).transpose(2, 3, 1, 0)}
            block_params["bn1"], block_stats["bn1"] = map_bn(base + "bn1.")
            block_params["bn2"], block_stats["bn2"] = map_bn(base + "bn2.")
            block_params["bn3"], block_stats["bn3"] = map_bn(base + "bn3.")
            if f"{base}downsample.0.weight" in state_dict or f"visual.{base}downsample.0.weight" in state_dict:
                block_params["down_conv"] = {
                    "kernel": _to_array(g(base + "downsample.0.weight")).transpose(2, 3, 1, 0),
                }
                block_params["down_bn"], block_stats["down_bn"] = map_bn(base + "downsample.1.")
            params[f"block_{block_idx}"] = block_params
            batch_stats[f"block_{block_idx}"] = block_stats
            block_idx += 1

    # AttentionPool2d
    embed_dim = cfg.width * 32
    head_dim = embed_dim // cfg.heads
    params["attnpool"] = {
        "positional_embedding": _to_array(g("attnpool.positional_embedding")),
        "SelfAttention_0": {
            "query": {
                "kernel": _to_array(g("attnpool.q_proj.weight")).T.reshape(embed_dim, cfg.heads, head_dim),
                "bias": _to_array(g("attnpool.q_proj.bias")).reshape(cfg.heads, head_dim),
            },
            "key": {
                "kernel": _to_array(g("attnpool.k_proj.weight")).T.reshape(embed_dim, cfg.heads, head_dim),
                "bias": _to_array(g("attnpool.k_proj.bias")).reshape(cfg.heads, head_dim),
            },
            "value": {
                "kernel": _to_array(g("attnpool.v_proj.weight")).T.reshape(embed_dim, cfg.heads, head_dim),
                "bias": _to_array(g("attnpool.v_proj.bias")).reshape(cfg.heads, head_dim),
            },
            "out": {
                "kernel": _to_array(g("attnpool.c_proj.weight")).T.reshape(cfg.heads, head_dim, cfg.output_dim),
                "bias": _to_array(g("attnpool.c_proj.bias")),
            },
        }
    }
    return params, batch_stats


def convert_clip_state_dict(state_dict: Dict[str, Any], clip_cfg: ClipCfg):
    """Map ref_clip CLIP state_dict to Flax CLIP params (vision + text + logit_scale).

    Returns (params, batch_stats) where batch_stats is nested under the 'vision' module
    for ResNet backbones (empty dict for ViT).
    """
    # vision
    if isinstance(clip_cfg.vision_cfg, VisionCfg):
        vision_params = convert_vit_state_dict(state_dict, clip_cfg.vision_cfg)
        vision_stats = {}
    else:
        vision_params, vision_stats = convert_resnet_state_dict(state_dict, clip_cfg.vision_cfg)
    # text shares weights directly
    text_params = convert_text_state_dict(state_dict, clip_cfg.text_cfg)

    params = {
        "vision": vision_params,
        "text": text_params,
        "logit_scale": _to_array(state_dict["logit_scale"]),
    }
    batch_stats = {"vision": vision_stats} if vision_stats else {}
    return params, batch_stats