import os, datetime, cv2, math, time, gc, torch
from os import path
from PIL import Image, ImageDraw
from lib.compute import GPU
from lib.camera import Camera
from lib.story import Storyteller
from lib.latent_image import LatentImage
from lib.state import save_loop, store_image

camera = Camera()
story = Storyteller()

def main_loop(loop_index, prev_step_image):

    story_latents = LatentImage()
    primer_latent_image = LatentImage()

    story_latents.from_image(prev_step_image)

    # img = story_latents.to_image()
    # store_image(img, 'base.png')

    # get latent space for camera view
    primer_latent_image.from_image(camera.grab_image(loop_index))
    img = primer_latent_image.to_image()
    # store_image(img, f'primer/{loop_index}.png')

    story_latents.merge_with(primer_latent_image.latents, scale = 0.5)
    img = story_latents.to_image()
    store_image(img, f'mixed/{loop_index}.png')

    story_latents.perturb(scale = 0.8)
    img = story_latents.to_image()
    store_image(img, f'perturbed/{loop_index}.png')

    # train the mixed latents on story
    # story.beat(loop_index)

    story_latents.from_text(story.to_embedding(), num_steps = 45)
    img = story_latents.to_image()
    store_image(img, f'final/{loop_index}.png')
    next_step_image = img

    # save_loop(index = loop_index, prompt = prompt, primer = primer_image, image = None )
    # time.sleep(get_wait_time())

    del story_latents
    del primer_latent_image
    torch.cuda.empty_cache()
    gc.collect()
    return next_step_image

def run_main_loop():
    loop_index = 100 # TODO persist to file

    init_latents = LatentImage()
    init_latents.from_text(story.to_embedding(), num_steps = 30)
    img = init_latents.to_image()
    del init_latents

    try:
        while loop_index < 110:
            print(' ')
            print(' ')
            print('loop', loop_index)
            GPU.getMemStats(f'pre loop: {loop_index}' )
            img = main_loop(loop_index, img)
            loop_index+=1
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

