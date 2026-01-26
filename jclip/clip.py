import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image

from .utils.logging_util import log_for_0, Emoji
from .utils.pjit_util import MeshMode

from .jax.model import ClipCfg, CLIP
from .jax.convert_weight import convert_clip_state_dict
from .jax.text_tower import TextCfg
from .jax.vision_tower import VisionCfg
from .torch.clip import load as load_torch_clip, tokenize as torch_tokenize, available_models

import time

def get_clip_cfg(model_name: str):
    assert model_name in ["ViT-B/32", "ViT-L/14"], f"Model {model_name} not supported."
    if model_name == "ViT-B/32":
        return ClipCfg(
            embed_dim=512,
            vision_cfg=VisionCfg(
                image_size=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                mlp_ratio=4.0,
                cls_token=True,
                output_dim=512,
            ),
            text_cfg=TextCfg(
                context_length=77,
                vocab_size=49408,
                width=512,
                heads=8,
                layers=12,
                cls_emb=False,
            ),
        )
    elif model_name == "ViT-L/14":
        return ClipCfg(
            embed_dim=768,
            vision_cfg=VisionCfg(
                image_size=224,
                patch_size=14,
                width=1024,
                layers=24,
                heads=16,
                mlp_ratio=4.0,
                cls_token=True,
                output_dim=768,
            ),
            text_cfg=TextCfg(
                context_length=77,
                vocab_size=49408,
                width=768,
                heads=12,
                layers=12,
                cls_emb=False,
            ),
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
def recursive_compare(param1, param2, name_str, success_log=False, until_fail=True):
    if not isinstance(param1, dict):
        if isinstance(param2, dict):
            print(f"!!!!! Key {name_str} is a leaf in param1 with shape {param1.shape} but not in param2")
            if until_fail:
                assert False, f"Key {name_str} is a leaf in param1 with shape {param1.shape} but not in param2"
            return
        if param1.shape != param2.shape:
            print(f"!!!!! Value {param1.shape} is not equal to param {name_str}.shape {param2.shape}")
            if until_fail:
                assert False, f"Value {param1.shape} is not equal to param {name_str}.shape {param2.shape}"
            return
        if success_log:
            print(f"[GOOD] Value {param1.shape} is equal to param {name_str}.shape {param2.shape}")
        return
    for key in param1.keys():
        if key not in param2:
            print(f"!!!!! Key {name_str}.{key} is in param1 but not in param2")
            if until_fail:
                assert False, f"Key {name_str}.{key} is in param1 but not in param2"
            continue
        recursive_compare(param1[key], param2[key], name_str + "." + key, success_log, until_fail)
    for key in param2.keys():
        if key not in param1:
            print(f"!!!!! Key {name_str}.{key} is in param2 but not in param1")
            if until_fail:
                assert False, f"Key {name_str}.{key} is in param2 but not in param1"
        
def test_consistency_with_torch(model_name: str):
    print("=" * 50)
    print(f"Testing consistency: {model_name}")
    print("=" * 50)
    
    # load torch model
    supported_models = available_models()
    assert model_name in supported_models, \
        f"Model {model_name} not supported in CLIP! Supported models: {supported_models}"
    model, preprocess = load_torch_clip(model_name, device="cpu", jit=False)
    state_dict = model.state_dict()
    
    # get jax clip config, and convert state dict to jax params
    clip_cfg = get_clip_cfg(model_name)
    params, batch_stats = convert_clip_state_dict(state_dict, clip_cfg)
    p_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("torch state_dict converted to jax params. param count:", p_count)
    
    # prepare dummy inputs
    tokens = torch_tokenize(["a diagram", "a dog", "a cat"], context_length=77)
    image_path = "/kmh-nfs-ssd-us-mount/code/xianbang/files/CLIP.png"
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0)  # add batch dimension
    image_input_jax = image_input.permute(0, 2, 3, 1)  # to NHWC
    
    # init jax model
    model_jax = CLIP(clip_cfg)
    img_np = np.array(image_input_jax)
    txt_np = tokens.numpy()
    rng = jax.random.PRNGKey(0)
    init_vars = model_jax.init(rng, img_np, txt_np, method=model_jax.__call__)["params"]
    jax_p_count = sum(x.size for x in jax.tree_util.tree_leaves(init_vars))
    print("param count for init_vars:", jax_p_count)
    
    # sanity tests
    assert p_count == jax_p_count, "param count mismatch"
    recursive_compare(params, init_vars, "params")
    
    # jax model forward
    img_features, text_features, _ = model_jax.apply({'params': params}, img_np, txt_np)
    print("JAX image features shape:", img_features.shape)
    print("JAX text features shape:", text_features.shape)
    print("JAX image features:", img_features[0, :10])
    print("JAX text features:", text_features[0, :10])
    
    # torch model forward
    with torch.no_grad():
        img_features_torch = model.encode_image(image_input)
        text_features_torch = model.encode_text(tokens)
    print("Torch image features shape:", img_features_torch.shape)
    print("Torch text features shape:", text_features_torch.shape)
    print("Torch image features:", img_features_torch[0, :10])
    print("Torch text features:", text_features_torch[0, :10])
    
    img_features = np.asarray(img_features)
    text_features = np.asarray(text_features)
    img_features_torch = img_features_torch.numpy()
    text_features_torch = text_features_torch.numpy()
    
    # compare jax and torch features
    print(f"{Emoji.INFO} Image features max diff: {np.max(np.abs(img_features - img_features_torch))}")
    print(f"{Emoji.INFO} Text features max diff: {np.max(np.abs(text_features - text_features_torch))}")
    
def create_clip_encode_fn(
    model_name: str,
    mesh_bundle=None, # for pjit compile; None for no compile
    modality: str = "text", # "text" or "image"
    max_encoder_length: int = 77, # only used for text modality
):
    clip_cfg = get_clip_cfg(model_name)
    model, preprocess = load_torch_clip(model_name, device="cpu", jit=False)
    state_dict = model.state_dict()
    params, _ = convert_clip_state_dict(state_dict, clip_cfg)
    model_jax = CLIP(clip_cfg)
    
    if mesh_bundle is None: # no pjit compile
        log_for_0(f"{Emoji.ROCKET} Creating non-pjit encode function for modality='{modality}' ...")
        if modality == "text":
            def encode_fn(params, text_input):
                text_features = model_jax.apply({'params': params}, text_input, method=model_jax.encode_text)
                return text_features.reshape((-1, clip_cfg.embed_dim))
            jit_encode_fn = jax.jit(encode_fn)
            return jit_encode_fn, params, clip_cfg
        elif modality == "image":
            def encode_fn(params, image_input):
                image_features = model_jax.apply({'params': params}, image_input, method=model_jax.encode_image)
                return image_features.reshape((-1, clip_cfg.embed_dim))
            jit_encode_fn = jax.jit(encode_fn)
            return jit_encode_fn, params, clip_cfg, preprocess
    
    log_for_0(f"{Emoji.ROCKET} Creating pjit encode function for modality='{modality}' ...")
    
    # get params spec
    tpu_mesh, get_partition_spec, _, _, pjit_compile = mesh_bundle
    params_spec = get_partition_spec(params, param_mode=MeshMode.MODEL)
    
    # shard params on mesh
    shard_params = pjit_compile(lambda x: x, in_shardings=(None,), out_shardings=params_spec)
    params = shard_params(params)
    
    # get data spec
    mesh_batch = int(np.prod(tpu_mesh.devices.shape))
    data_shape = {
        "image_input": jax.ShapeDtypeStruct((mesh_batch, 224, 224, 3), jnp.float32), # not used for now
        "text_input": jax.ShapeDtypeStruct((mesh_batch, max_encoder_length), jnp.int32),
    }
    data_spec = get_partition_spec(data_shape, param_mode=MeshMode.DATA)
    output_spec = get_partition_spec(
        jax.ShapeDtypeStruct((mesh_batch, clip_cfg.embed_dim), jnp.float32),
        param_mode=MeshMode.DATA,
    )
    
    if modality == "text":
        def _encode_fn(params, text_input):
            text_features = model_jax.apply({'params': params}, text_input, method=model_jax.encode_text)
            return text_features.reshape((-1, clip_cfg.embed_dim))
        
        pjit_encode_fn = pjit_compile(
            _encode_fn,
            in_shardings=(params_spec, data_spec["text_input"]),
            out_shardings=output_spec,
        )
        
        return pjit_encode_fn, params, clip_cfg
    elif modality == "image":
        def _encode_fn(params, image_input):
            image_features = model_jax.apply({'params': params}, image_input, method=model_jax.encode_image)
            return image_features.reshape((-1, clip_cfg.embed_dim))
        
        pjit_encode_fn = pjit_compile(
            _encode_fn,
            in_shardings=(params_spec, data_spec["image_input"]),
            out_shardings=output_spec,
        )
        
        return pjit_encode_fn, params, clip_cfg, preprocess

class CLIPTokenizer:
    def __init__(self, context_length: int = 77):
        self.context_length = context_length
        self.tokenizer = torch_tokenize
    
    def tokenize_single(self, text: str):
        return self.tokenizer([text], context_length=self.context_length, truncate=True)
    
    def tokenize_batch(self, texts: list[str]):
        tokenized = self.tokenizer(texts, context_length=self.context_length, truncate=True)
        return np.stack([x.numpy() for x in tokenized], axis=0)

if __name__ == "__main__":
    # ---> test convert weight consistency with torch
    # test_consistency_with_torch("ViT-B/32")
    # test_consistency_with_torch("ViT-L/14")
    
    # ---> test text encode fn
    # encode_fn, params, clip_cfg = create_clip_encode_fn(
    #     "ViT-L/14",
    #     modality="text",
    # )
    # print("encode_fn and params prepared.")
    # print("params count:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    # # test forward
    # text_input = np.random.randint(0, 1000, (8, 77)).astype(np.int32)
    # text_features = encode_fn(params, text_input)
    # print("text_features shape:", text_features.shape)
    # print("text_features:", text_features[0, :10])
    
    # ---> test image encode fn
    encode_fn, params, clip_cfg, preprocess = create_clip_encode_fn(
        "ViT-L/14",
        modality="image",
    )
    print("encode_fn and params prepared.")
    print("params count:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    # test forward
    image_input = np.random.randn(8, 224, 224, 3).astype(np.float32)
    image_features = encode_fn(params, image_input)
    print("image_features shape:", image_features.shape)
    print("image_features:", image_features[0, :10])
    # test speed
    start_time = time.time()
    for _ in range(10):
        image_features = encode_fn(params, image_input)
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    print(f"Average inference time over 10 runs: {avg_time*1000:.2f} ms") # expect < 2ms