import os, datetime, cv2, math, time, gc, torch
import climage
from os import path
from PIL import Image, ImageDraw
from lib.compute import GPU
from lib.camera import Camera
from lib.story import Storyteller
from lib.latent_image import LatentImage
from lib.state import save_loop, store_image

camera = Camera()
story = Storyteller()
story_latents = LatentImage()
primer_latent_image = LatentImage()

def main_loop(loop_index):
    print(' ')
    print(' ')
    print('start', loop_index)
    GPU.getMemStats()

    # get latent space for camera view
    primer_latent_image.from_image(camera.grab_image(loop_index))
    # img = primer_latent_image.to_image()
    # store_image(img, f'primer/{loop_index}.png')

    story_latents.merge_with(primer_latent_image.latents, scale = 0.5)
    # img = story_latents.to_image()
    # store_image(img, f'mixed/{loop_index}.png')

    # train the mixed latents on story
    # story.beat(loop_index)
    text_embeddings = story.to_embedding()
    
    story_latents.perturb(scale = 0.4)
    # img = story_latents.to_image()
    # store_image(img, f'perturbed/{loop_index}.png')

    story_latents.from_text(text_embeddings, num_steps = 45)
    img = story_latents.to_image()
    store_image(img, f'final/{loop_index}.png')

    # save_loop(index = loop_index, prompt = prompt, primer = primer_image, image = None )
    # time.sleep(get_wait_time())

def run_main_loop():
    loop_index = 100 # TODO persist to file

    story_latents.from_text(story.to_embedding(), num_steps = 30)
    img = story_latents.to_image()
    store_image(img, 'base.png')

    try:
        while loop_index < 110:
            print('loop', loop_index)
            main_loop(loop_index)
            loop_index+=1
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        print('Clean')
        GPU.clean()

# def get_rate_of_change(loop_index):
#     sin = round((math.sin(loop_index) +1 ) /2, 4)
#     cos = round((math.cos(loop_index) +1 ) /2, 4)
#     return [sin, cos]
# # TODO
# def get_image_from_frame(frame):
#     pass #upscale
# def save_interval(index, prompt, frame, primer):
#     timestamp = datetime.datetime.now().isoformat()
#     # cv2.imwrite(work_path + f"frame_{loop_index}_{prompt.replace(' ', '_')}.png", frame)
#     cv2.imwrite(work_path + f"primer_{timestamp}.png", primer)
# # // to regulate FPS
# def get_wait_time():
#     # currently: # after previous loop finished and second/x passed.
#     # future: # later may be controlled by performance monitor
#     return 5000/1000



run_main_loop()

# cv2.destroyAllWindows()

