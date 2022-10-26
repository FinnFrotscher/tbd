import numpy as np
import torch

from lib.model import GPU

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

    latent_dists = GPU.vae.encode(img_arr.to(GPU.device)).latent_dist

    latent_samples = latent_dists.sample()
    latent_samples *= 0.18215

    return latent_samples[0]


