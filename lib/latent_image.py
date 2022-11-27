import argparse, os, sys, glob, cv2, gc
import numpy as np
import torch
from globals import *
from lib.image import Image
from os import path
from tqdm.auto import tqdm
from torch import autocast
from torch.nn import functional as F
from lib.compute import GPU

class LatentImage:
    latents = None
    height = output_dimensions
    width = output_dimensions
    guidance_scale = 7.5

    def from_text(self, text_embeddings, num_steps = 50, start_step = 0):
        if self.latents is None:
            self.latents = torch.randn((text_embeddings.shape[0] // 2, GPU.unet.in_channels, self.height // 8, self.width // 8))

        self.latents = self.latents.to(GPU.device)

        if start_step > 0:
            scheduler_type = "DDIMscheduler"
            scheduler = GPU.DDIMscheduler
        # else:
        scheduler_type = "LMSDscheduler"
        scheduler = GPU.LMSDscheduler
        print('scheduler_type', scheduler_type)

        scheduler.set_timesteps(num_steps)

        if scheduler_type == "LMSDscheduler":
            self.latents = self.latents * scheduler.sigmas[0]

        if scheduler_type == "DDIMscheduler":
            start_timestep = scheduler.timesteps[start_step]
            start_timesteps = start_timestep.repeat(self.latents.shape[0]).long()
            noise = torch.randn_like(self.latents)
            latents = scheduler.add_noise(self.latents, noise, start_timesteps)

        with autocast(GPU.device):
            for i, t in tqdm(enumerate(scheduler.timesteps[start_step:])):
                if scheduler_type == "LMSDscheduler":
                    scheduler_index = i
                else:
                    scheduler_index = t

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([self.latents] * 2)

                if scheduler_type == "LMSDscheduler":
                    sigma = scheduler.sigmas[i]
                    latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = GPU.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                self.latents = scheduler.step(noise_pred, scheduler_index, self.latents)['prev_sample']

    def perturb(self, scale = 0):
        noise = torch.randn_like(self.latents)
        new_latents = (1 - scale) * self.latents + scale * noise
        self.latents =  (new_latents - new_latents.mean()) / new_latents.std()

    def merge_with(self, input_latents, scale = 0.5):
        new_latents = (1 - scale) * self.latents + scale * input_latents
        self.latents =  (new_latents - new_latents.mean()) / new_latents.std()

    def to_image(self):
        scaled_latents = 1 / 0.18215 * self.latents

        with torch.no_grad():
            imgs = GPU.vae.decode(scaled_latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        return Image(imgs[0])

    def from_image(self, image):
        if not isinstance(image, list):
            imgs = [image.raw]

        img_arr = np.stack([ np.array(img) for img in imgs ], axis=0)
        img_arr = img_arr / 255.0

        # probably have to remove a dimension here
        img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)

        latent_dists = GPU.vae.encode(img_arr.to(GPU.device)).latent_dist
        latent_samples = latent_dists.sample()
        latent_samples *= 0.18215

        self.latents = latent_samples







