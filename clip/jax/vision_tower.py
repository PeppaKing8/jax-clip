from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from .text_tower import ResidualAttentionBlock, feature_take_indices

@dataclass
class VisionCfg:
    image_size: int = 224
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    output_dim: int = 512
    pool_type: str = "tok"  # tok/avg/none/last
    output_tokens: bool = False
    cls_token: bool = True
    patch_dropout: float = 0.0
    no_ln_pre: bool = False

@dataclass
class ResNetCfg:
    layers: List[int]
    output_dim: int
    heads: int
    image_size: int = 224
    width: int = 64


class PatchDropout(nn.Module):
    prob: float = 0.0
    exclude_first_token: bool = True

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        if deterministic or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls, rest = x[:, :1], x[:, 1:]
        else:
            cls = None
            rest = x
        B, N, C = rest.shape
        keep = jax.random.bernoulli(self.make_rng("dropout"), 1.0 - self.prob, (B, N))
        keep_indices = jnp.argsort(keep, axis=1)[:, ::-1]
        num_keep = jnp.maximum(1, (keep.sum(axis=1)))
        idx = jnp.arange(keep_indices.shape[1])
        idx = idx[None, :] < num_keep[:, None]
        keep_indices = jnp.where(idx, keep_indices, keep_indices[:, :1])
        rest = jnp.take_along_axis(rest, keep_indices[:, :, None], axis=1)
        if self.exclude_first_token and cls is not None:
            rest = jnp.concatenate([cls, rest], axis=1)
        return rest


class VisionTransformer(nn.Module):
    cfg: VisionCfg

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.cfg.width,
            kernel_size=(self.cfg.patch_size, self.cfg.patch_size),
            strides=(self.cfg.patch_size, self.cfg.patch_size),
            padding="VALID",
            use_bias=False,
            name="conv1",
        )
        num_patches = (self.cfg.image_size // self.cfg.patch_size) ** 2
        pos_tokens = num_patches + (1 if self.cfg.cls_token else 0)
        self.positional_embedding = self.param(
            "positional_embedding", nn.initializers.normal(stddev=0.01), (pos_tokens, self.cfg.width)
        )
        if self.cfg.cls_token:
            self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.cfg.width))
        else:
            self.cls_token = None
        self.ln_pre = nn.Identity() if self.cfg.no_ln_pre else nn.LayerNorm(epsilon=1e-5, name="ln_pre")
        self.blocks = [
            ResidualAttentionBlock(
                dim=self.cfg.width,
                num_heads=self.cfg.heads,
                mlp_ratio=self.cfg.mlp_ratio,
                attn_drop=0.0,
                proj_drop=0.0,
                name=f"resblocks_{i}",
            )
            for i in range(self.cfg.layers)
        ]
        self.ln_post = nn.LayerNorm(epsilon=1e-5, name="ln_post")
        self.proj = self.param("proj", nn.initializers.normal(stddev=self.cfg.width ** -0.5), (self.cfg.width, self.cfg.output_dim))

    def _pool(self, x):
        if self.cfg.pool_type == "avg":
            pooled = x[:, 1:].mean(axis=1)
            tokens = x[:, 1:]
        elif self.cfg.pool_type == "tok":
            pooled = x[:, 0]
            tokens = x[:, 1:]
        elif self.cfg.pool_type == "last":
            pooled = x[:, -1]
            tokens = x[:, :-1]
        else:  # none
            pooled = x
            tokens = x
        pooled = pooled @ self.proj
        return pooled, tokens

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = True,
        intermediates: bool = False,
        intermediate_indices: Optional[Union[int, List[int]]] = None,
    ):
        # x: [B, H, W, 3] NHWC
        x = self.conv1(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)

        # print("x after conv1:", x[0, :5, :5])

        if self.cls_token is not None:
            cls_tok = jnp.broadcast_to(self.cls_token, (B, 1, C))
            x = jnp.concatenate([cls_tok, x], axis=1)

        x = x + self.positional_embedding[None, : x.shape[1], :]
        x = self.ln_pre(x)
        
        # print("x after ln_pre:", x[0, :5, :5])

        # take_indices, max_index = feature_take_indices(len(self.blocks), intermediate_indices)
        take_indices = []
        max_index = None
        ints: List[jnp.ndarray] = []
        blocks = self.blocks[: max_index + 1] if max_index is not None else self.blocks
        for i, blk in enumerate(blocks):
            x = blk(x, attn_mask=None, deterministic=deterministic)
            if i in take_indices:
                ints.append(x)
            # print(f"x after block {i}:", x[0, :5, :5])

        x = self.ln_post(x)
        # print("x after ln_post:", x[0, :5, :5])
        pooled, tokens = self._pool(x)
        # print("pooled:", pooled[0, :5])
        if self.cfg.output_tokens or intermediates:
            out: Dict[str, Union[jnp.ndarray, List[jnp.ndarray]]] = {
                "image_features": pooled,
                "image_tokens": tokens,
            }
            if intermediates:
                out["image_intermediates"] = ints
            return out
        return pooled


class Bottleneck(nn.Module):
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, *, train: bool):
        in_channels = x.shape[-1]
        conv1 = nn.Conv(self.out_channels, (1, 1), strides=(1, 1), use_bias=False, name="conv1")
        bn1 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="bn1")
        conv2 = nn.Conv(self.out_channels, (3, 3), strides=(1, 1), padding="SAME", use_bias=False, name="conv2")
        bn2 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="bn2")
        conv3 = nn.Conv(self.out_channels * 4, (1, 1), strides=(1, 1), use_bias=False, name="conv3")
        bn3 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="bn3")

        needs_downsample = self.stride > 1 or in_channels != self.out_channels * 4
        if needs_downsample:
            down_conv = nn.Conv(self.out_channels * 4, (1, 1), strides=(1, 1), use_bias=False, name="down_conv")
            down_bn = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="down_bn")

        identity = x

        out = conv1(x)
        out = bn1(out, use_running_average=not train)
        out = nn.relu(out)

        out = conv2(out)
        out = bn2(out, use_running_average=not train)
        out = nn.relu(out)

        if self.stride > 1:
            out = nn.avg_pool(out, window_shape=(self.stride, self.stride), strides=(self.stride, self.stride), padding="VALID")

        out = conv3(out)
        out = bn3(out, use_running_average=not train)

        if needs_downsample:
            identity = nn.avg_pool(identity, window_shape=(self.stride, self.stride), strides=(self.stride, self.stride), padding="VALID")
            identity = down_conv(identity)
            identity = down_bn(identity, use_running_average=not train)

        out = nn.relu(out + identity)
        return out


class AttentionPool2D(nn.Module):
    spacial_dim: int
    embed_dim: int
    num_heads: int
    output_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x, *, train: bool):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        tokens = x.reshape(B, H * W, C)
        cls_token = tokens.mean(axis=1, keepdims=True)
        tokens = jnp.concatenate([cls_token, tokens], axis=1)  # [B, HW+1, C]

        pos_emb = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=self.embed_dim ** -0.5),
            (self.spacial_dim ** 2 + 1, self.embed_dim),
        )
        tokens = tokens + pos_emb[None, :, :]

        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=not train,
            out_features=self.output_dim or self.embed_dim,
        )
        out = attn(tokens)
        out = out[:, 0]
        return out


class ModifiedResNet(nn.Module):
    cfg: ResNetCfg

    def setup(self):
        width = self.cfg.width
        # stem
        self.conv1 = nn.Conv(width // 2, (3, 3), strides=(2, 2), padding="SAME", use_bias=False, name="conv1")
        self.bn1 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="bn1")
        self.conv2 = nn.Conv(width // 2, (3, 3), padding="SAME", use_bias=False, name="conv2")
        self.bn2 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="bn2")
        self.conv3 = nn.Conv(width, (3, 3), padding="SAME", use_bias=False, name="conv3")
        self.bn3 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, name="bn3")

        # layers (flattened for stable naming)
        planes_list = [width, width * 2, width * 4, width * 8]
        strides = [1, 2, 2, 2]
        blocks = []
        idx = 0
        for planes, blocks_num, stride in zip(planes_list, self.cfg.layers, strides):
            for b in range(blocks_num):
                blk_stride = stride if b == 0 else 1
                blocks.append(Bottleneck(planes, stride=blk_stride, name=f"block_{idx}"))
                idx += 1
        self.blocks = blocks

        embed_dim = width * 32
        self.attnpool = AttentionPool2D(self.cfg.image_size // 32, embed_dim, self.cfg.heads, self.cfg.output_dim, name="attnpool")

    def __call__(
        self,
        x,
        *,
        train: bool = True,
        intermediates: bool = False,
        intermediate_indices: Optional[Union[int, List[int]]] = None,
    ):
        # x: [B, H, W, 3] NHWC
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")

        take_indices, max_index = feature_take_indices(len(self.blocks), intermediate_indices)
        ints: List[jnp.ndarray] = []
        blocks = self.blocks if max_index is None else self.blocks[: max_index + 1]
        for i, block in enumerate(blocks):
            x = block(x, train=train)
            if i in take_indices:
                ints.append(x)

        x = self.attnpool(x, train=train)
        if intermediates:
            return {"image_features": x, "image_intermediates": ints}
        return x