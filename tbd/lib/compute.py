import argparse, os, sys, glob, cv2, gc
import torch
from os import path
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

device = 'cuda'
cwd = path.join(os.getcwd())
modelpath = 'models/ldm/stable-diffusion-v1'
loadpath = path.normpath(path.join(cwd, '..', modelpath))

class GPUHandler:
    device = 'cuda'
    vae = None
    unet = None
    scheduler = None
    tokenizer = None
    text_encoder = None

    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained('/home/finn/data/stable-diffusion-v1-4/vae')
        self.unet = UNet2DConditionModel.from_pretrained('/home/finn/data/stable-diffusion-v1-4/unet')
        self.tokenizer = CLIPTokenizer.from_pretrained('/home/finn/data/stable-diffusion-v1-4/tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained('/home/finn/data/stable-diffusion-v1-4/text_encoder')

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000
        )

        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)


    def clean(self):
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.tokenizer = None
        self.text_encoder = None

        torch.cuda.empty_cache()
        gc.collect()

GPU = GPUHandler()
