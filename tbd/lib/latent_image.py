import argparse, os, sys, glob, cv2, gc
from os import path
from tqdm.auto import tqdm
from base64 import b64encode
from PIL import Image, ImageDraw
from IPython.display import HTML
import numpy as np
import torch
from torch import autocast, float16
from torch.nn import functional as F

from lib.compute import GPU

class LatentImage:
    height = 512
    width = 512
    guidance_scale = 7.5

    def __init__(self):
        self.latents = torch.randn((
            1, GPU.unet.in_channels,
            512 // 8, 512 // 8
        ))

    def produce_latents(
            self,
            text_embeddings,
            num_inference_steps = 50,
            start_step = 0
    ):
        temp_latents = self.latents.to(GPU.device)

        GPU.scheduler.set_timesteps(num_inference_steps)

        if start_step > 0:
            start_timestep = GPU.scheduler.timesteps[start_step]
            start_timesteps = start_timestep.repeat(temp_latents.shape[0]).long()

            noise = torch.randn_like(temp_latents)
            latents = scheduler.add_noise(temp_latents, noise, start_timesteps)

        with autocast('cuda'):
            for i, t in tqdm(enumerate(GPU.scheduler.timesteps[start_step:])):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([temp_latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = GPU.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                temp_latents = GPU.scheduler.step(noise_pred, t, temp_latents)['prev_sample']

        self.latents = temp_latents

    def perturb(self, scale = 0):
        noise = torch.randn_like(self.latents)
        new_latents = (1 - scale) * self.latents + scale * noise
        self.latents =  (new_latents - new_latents.mean()) / new_latents.std()

    def merge_with_latents(self, input_latents, scale = 0):
        # ??

    def to_image(self):
        scaled_latents = 1 / 0.18215 * self.latents

        with torch.no_grad():
            img = GPU.vae.decode(scaled_latents).sample

        # probably have to remove a dimension here
        img = (img + 0.5).clamp(0, 1)
        img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype('uint8')

        return Image.fromarray(img)

    def from_image(self, image):
        img = np.stack(np.array(image), axis=0)
        img = img / 255.0

        # probably have to remove a dimension here
        img_arr = torch.from_numpy(img).float().permute(0, 3, 1, 2)
        img_arr = 2 * (img - 0.5)

        latent_dists = GPU.vae.encode(img.to(GPU.device)).latent_dist

        latent_sample = latent_dists.sample()
        latent_sample *= 0.18215

        self.latents = latent_samples




latents = LatentImage()




