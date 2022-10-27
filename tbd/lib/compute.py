import argparse, os, sys, glob, cv2, gc
import torch
from pynvml import *
from os import path
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


device = 'cuda'
modelpath = '/home/finn/data/stable-diffusion-v1-4/'

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)

class GPUHandler:
    device = 'cuda'
    vae = None
    unet = None
    scheduler = None
    tokenizer = None
    text_encoder = None

    def __init__(self):
        print('init')
        self.getMemStats()
        self.vae = AutoencoderKL.from_pretrained(path.join(modelpath, 'vae'))
        self.unet = UNet2DConditionModel.from_pretrained(path.join(modelpath, 'unet'))
        self.tokenizer = CLIPTokenizer.from_pretrained(path.join(modelpath, 'tokenizer'))
        self.text_encoder = CLIPTextModel.from_pretrained(path.join(modelpath, 'text_encoder'))

        # self.LMSDscheduler = LMSDiscreteScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule='scaled_linear',
        #     num_train_timesteps=1000
        # )

        self.DDIMscheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000
        )
        # self.scheduler = self.LMSDscheduler
        self.scheduler = self.DDIMscheduler

        print('prior to x.to(device)')
        self.getMemStats()
        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        print('After x.to(device)')
        self.getMemStats()


    def getMemStats(self):
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'free: {info.free // 1024 ** 2}mb  | used: {info.used // 1024 ** 2}mb')

    def clean(self):
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.tokenizer = None
        self.text_encoder = None

        torch.cuda.empty_cache()
        gc.collect()

GPU = GPUHandler()
