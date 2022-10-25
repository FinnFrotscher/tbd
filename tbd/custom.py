import argparse, os, sys, glob, cv2
from os import path
from base64 import b64encode
from PIL import Image, ImageDraw
from IPython.display import HTML

import numpy as np
import torch
from torch import autocast
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

# from omegaconf import OmegaConf
# from tqdm import tqdm, trange
# from itertools import islice
# from einops import rearrange, repeat
# from torchvision.utils import make_grid
# from contextlib import nullcontext
# import time
# from pytorch_lightning import seed_everything

# from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler
# from ldm.dream.devices         import choose_torch_device

device = 'cuda'

cwd = path.join(os.getcwd())
modelpath = 'models/ldm/stable-diffusion-v1'
loadpath = path.normpath(path.join(cwd, '..', modelpath))

# print(loadpath)

vae = AutoencoderKL.from_pretrained('/home/finn/data/stable-diffusion-v1-4/vae')
vae.to(device)

tokenizer = CLIPTokenizer.from_pretrained('/home/finn/data/stable-diffusion-v1-4/tokenizer')
text_encoder = CLIPTextModel.from_pretrained('/home/finn/data/stable-diffusion-v1-4/text_encoder')
text_encoder.to(device)


unet = UNet2DConditionModel.from_pretrained('/home/finn/data/stable-diffusion-v1-4/unet')
unet.to(device)

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule='scaled_linear',
    num_train_timesteps=1000
)


def get_text_embeds(prompt):
    text_input = tokenizer (
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        tuncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = tokenizer(
        [''] * len(prompt),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        tuncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


get_text_embeds('lorem ipsum')
