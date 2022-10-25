import argparse, os, sys, glob, cv2
from os import path
from tqdm.auto import tqdm
from base64 import b64encode
from PIL import Image, ImageDraw
from IPython.display import HTML

import numpy as np
import torch
from torch import autocast, float16
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

device = 'cuda'
cwd = path.join(os.getcwd())
modelpath = 'models/ldm/stable-diffusion-v1'
loadpath = path.normpath(path.join(cwd, '..', modelpath))


# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained('/home/finn/data/stable-diffusion-v1-4/vae')
vae.to(device)

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained('/home/finn/data/stable-diffusion-v1-4/unet')
unet.to(device)


# 4. Create a scheduler for inference
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule='scaled_linear',
    num_train_timesteps=1000
)


# Make a Video
# prompt = 'Starry night with a violet sky digital art'
# video_frames = prompt_to_img(prompt, num_inference_steps=40, return_all_latents=True)

# Similar Img
# prompt = 'Steampunk airship bursting through the clouds, cyberpunk art'
# latents = torch.randn((1, unet.in_channels, 512 // 8, 512 // 8))
# img = prompt_to_img(prompt, num_inference_steps=20, latents=latents)[0]
# #
# new_latents = perturb_latents(latents, 0.4)
# img = prompt_to_img(prompt, num_inference_steps=20, latents=new_latents)[0]

# Img-to-Img
# img = prompt_to_img(prompt, num_inference_steps=30)[0]
# img_latents = encode_img_latents([img])
def load_first_latents(): # maybe with embedding?
    latents = torch.randn((1, unet.in_channels, 512 // 8, 512 // 8))
    # latents = torch.randn((
    #     text_embeddings.shape[0] // 2,
    #     unet.in_channels,
    #     height // 8,
    #     width // 8
    # ))
    return latents

def produce_latents(
        text_embeddings,
        latents,
        height=512, width=512,
        num_inference_steps=50,
        start_step=10,
        guidance_scale=7.5
):
    latents = latents.to(device)

    scheduler.set_timesteps(num_inference_steps)

    if start_step > 0:
        start_timestep = scheduler.timesteps[start_step]
        start_timesteps = start_timestep.repeat(latents.shape[0]).long()

        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, start_timesteps)

    with autocast('cuda'):
        for i, t in tqdm(enumerate(scheduler.timesteps[start_step:])):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']

    return latents


# def prompt_to_img(prompts, height=512, width=512, num_inference_steps=50,
#                   guidance_scale=7.5, latents=None, return_all_latents=False,
#                   batch_size=2, start_step=0):



def perturb_latents(latents, scale=0.1):
  noise = torch.randn_like(latents)
  new_latents = (1 - scale) * latents + scale * noise
  return (new_latents - new_latents.mean()) / new_latents.std()

# # for image to image
# probably not needed because I save latents between rounds
# possible useful to run from a specific starter image
# or if i figure out how to join latent spaces -to> to prime the main latents with the primer image
def image_to_latents(imgs):
  if not isinstance(imgs, list):
    imgs = [imgs]

  img_arr = np.stack([np.array(img) for img in imgs], axis=0)
  img_arr = img_arr / 255.0
  img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
  img_arr = 2 * (img_arr - 0.5)

  latent_dists = vae.encode(img_arr.to(device))
  latent_samples = latent_dists.sample()
  latent_samples *= 0.18215

  return latent_samples[0]

def latents_to_image(latents):
  # if not isinstance(latents, list):
  #   latents = [latents]
  latents = 1 / 0.18215 * latents

  with torch.no_grad():
    imgs = vae.decode(latents).sample

  imgs = (imgs + 0.5).clamp(0, 1)
  imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
  imgs = (imgs * 255).round().astype('uint8')
  pil_images = [Image.fromarray(image) for image in imgs]
  return pil_images

