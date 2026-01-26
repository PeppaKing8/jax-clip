import os
import torch
import torch.nn as nn

def get_aesthetic_head(
    clip_model_name: str = "ViT-L/14",
    cache_path: str = "/kmh-nfs-ssd-us-mount/code/xianbang/files/",
):
    assert clip_model_name in ["ViT-B/32", "ViT-L/14"], f"Model {clip_model_name} not supported."
    
    path = os.path.expanduser(cache_path)
    path_to_model = path + f'''sa_0_4_{
        clip_model_name.lower().replace('/', '_').replace('-', '_')
    }_linear.pth'''
    assert os.path.exists(path), f"Cache path {path} does not exist. Please create it first."
    
    if clip_model_name == "ViT-L/14":
        m = nn.Linear(768, 1)
    elif clip_model_name == "ViT-B/32":
        m = nn.Linear(512, 1)
        
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m