# jax-clip
OpenAI CLIP encoders and an aesthetic head implemented in JAX/Flax.

- [Loading CLIP image encoder](#loading-clip-image-encoder)
- [Evaluating aesthetic score for an image](#evaluating-aesthetic-score-for-an-image)

## Loading CLIP image encoder
```python
import jax.numpy as jnp
from PIL import Image
from jclip import create_clip_encode_fn

# Build encoder, params, and preprocess
clip_fn, clip_params, _, preprocess = create_clip_encode_fn(
	model_name="ViT-L/14",
	modality="image",
)

# Preprocess an image (CHW float32), convert to JAX array, and add batch dim
image = preprocess(Image.open("path/to/image.jpg"))
image = jnp.array(image).transpose(1, 2, 0)[None, ...]

# Encode
image_features = clip_fn(clip_params, image)
print(image_features.shape)
```

## Evaluating aesthetic score for an image

`get_aesthetic_fn` wraps CLIP with a learned 1-layer head, returning a callable and its parameters; see [aesthetic/aesthetic.py](aesthetic/aesthetic.py#L16-L57).

```python
from aesthetic.aesthetic import get_aesthetic_fn
from PIL import Image

aesthetic_fn, params, preprocess = get_aesthetic_fn(
	clip_model_name="ViT-L/14",
	get_preprocess=True,
)

image = Image.open("path/to/image.jpg")
image = preprocess(image) # prepare as jnp.array, HWC
image = image[None, ...]  # add batch dim

score = aesthetic_fn(params, image)
print(float(score[0]))
```