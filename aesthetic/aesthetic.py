import flax.linen as nn 
import jax
import jax.numpy as jnp

from .torch_head import get_aesthetic_head as get_head_torch
from clip.clip import create_clip_encode_fn

class AestheticHead(nn.Module):
    input_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1)(x)
        return x

def get_aesthetic_fn(
    clip_model_name: str = "ViT-L/14",
    head_cache_path: str = "/kmh-nfs-ssd-us-mount/code/xianbang/files/",
    get_preprocess: bool = False,
):
    head_torch = get_head_torch(
        clip_model_name=clip_model_name,
        cache_path=head_cache_path,
    )
    input_dim = 768 if clip_model_name == "ViT-L/14" else 512
    head_jax = AestheticHead(input_dim=input_dim)
    head_params = {
        'Dense_0': {
            'kernel': jnp.array(head_torch.weight.data.numpy().T),
            'bias': jnp.array(head_torch.bias.data.numpy()),
        }
    }
    clip_fn, clip_params, _, preprocess = create_clip_encode_fn(
        model_name=clip_model_name,
        modality="image",
    )
    params = {
        'clip': clip_params,
        'aesthetic_head': head_params,
    }
    
    def aesthetic_fn(params, image_input):
        image_features = clip_fn(params['clip'], image_input)
        scores = head_jax.apply({'params': params['aesthetic_head']}, image_features)
        return scores.reshape((-1,))
    aesthetic_fn = jax.jit(aesthetic_fn)
    
    if get_preprocess:
        def jax_preprocess(image):
            image = preprocess(image)
            image = jnp.array(image)
            image = image.transpose(1, 2, 0)  # CHW to HWC
            return image
        
        return aesthetic_fn, params, jax_preprocess
    else:
        return aesthetic_fn, params

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    model_name = "ViT-L/14"
    aesthetic_fn, params, preprocess = get_aesthetic_fn(
        clip_model_name=model_name,
        head_cache_path="/kmh-nfs-ssd-us-mount/code/xianbang/files/",
        get_preprocess=True,
    )
    
    image_path = "/kmh-nfs-ssd-us-mount/code/xianbang/files/samples_imgnet/1.JPEG"
    image = Image.open(image_path)
    image_input = preprocess(image)
    image_input = image_input[None, ...]  # Add batch dimension
    scores = aesthetic_fn(params, image_input)
    print("Aesthetic score for a real image:", scores)
    
    random_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    random_image = preprocess(Image.fromarray(random_image))
    random_image = random_image[None, ...]  # Add batch dimension
    scores = aesthetic_fn(params, random_image)
    print("Aesthetic score for a random image:", scores)
    
    # test speed
    import time
    start_time = time.time()
    for _ in range(100):
        scores = aesthetic_fn(params, random_image)
    end_time = time.time()
    print("Average inference time over 100 runs:", (end_time - start_time) / 100) # expect < 2ms