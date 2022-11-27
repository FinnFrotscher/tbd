import os, datetime, cv2, math, time, gc, torch
from os import path
from lib.compute import GPU
from lib.image import Image
from lib.camera import Camera
from lib.story import Storyteller
from lib.latent_image import LatentImage


def main_loop(loop_index, prev_step_image):
    camera = Camera()
    story = Storyteller()
    story_latents = LatentImage()
    # primer_latent_image = LatentImage()

    # prev_step_image.apply_brightness_contrast(0,0)
    story_latents.from_image(prev_step_image)
    del prev_step_image

    # primer_image = camera.grab_image(loop_index)
    # primer_image.apply_brightness_contrast()
    # primer_image.gray()

    # primer_latent_image.from_image(primer_image)

    # story_latents.merge_with(primer_latent_image.latents, scale = 0.2)
    # img = story_latents.to_image()
    # img.store(f'mixed/{loop_index}.png')

    story_latents.perturb(scale = 0.1)
    img = story_latents.to_image()
    img.store(f'perturbed/{loop_index}.png')

    # story.beat(loop_index)

    story_latents.from_text(story.to_embedding(), start_step = 40, num_steps = 80)
    img = story_latents.to_image()
    img.store(f'final/{loop_index}.png')
    next_step_image = img

    # save_loop(index = loop_index, prompt = prompt, primer = primer_image, image = None )
    # time.sleep(get_wait_time())

    del img
    del story_latents
    # del primer_latent_image
    del camera
    del story
    torch.cuda.empty_cache()
    gc.collect()
    return next_step_image

def run_main_loop():
    loop_index = 100 # TODO persist to file

    story = Storyteller()
    init_latents = LatentImage()
    init_latents.from_text(story.to_embedding(), num_steps = 80)
    img = init_latents.to_image()
    img.store(f'init.png')
    del init_latents
    del story

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

# image = Camera().grab_image(1005)
# image.apply_brightness_contrast()
# image.store('output.png')
