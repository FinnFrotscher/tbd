import datetime
import cv2
import math
import time

from diffusion.latents import get_latent_state, load_first_latents
from prompt.prompts import get_prompt

latents = load_first_latents()

cam = cv2.VideoCapture(0)
cv2.namedWindow("test2")

cwd = path.join(os.getcwd())
out_path = 'tbd/output'
work_dir = path.normpath(path.join(cwd, '..', out_path))


def main_loop(loop_index=0):
    [sin, cos] = get_rate_of_change(loop_index)
    prompt = get_prompt(loop_index)
    text_embeds = prompt_to_embed(prompts)

    primer_image = get_primer_image()
    primer_image_latents = image_to_latents(primer_image)

    # TODO
    # i have the current latents, the primer latents, the prompt embeddings
    # and a sin and cosin wave
    # i want out the latent space that was mutated by prompt embeddings

    # i can also blurr any latents  (by pertubing by scale X)
    # and deblurs towards the prompt (by num_inference_steps Y)

    # i will combine primer and current latents by matric multiplication
    # if that doesnt work i will have to merge the final image from the previous loop with the actual image from the camera with an aplpha effect and extract the latents of that.
    # i will blurr the combined latent space of the primer and the current latents

    main_latents = perturb_latents(previous_latents, sin)
    # how to multiply latent states?
    main_latents = main_latents * primer_image_latents # weigh this operation by rate_of_change.cos
    # latents = produce_latents(
    #     text_embeds = text_embeds,
    #     latents=latents,
    #     num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
    #     height=height, width=width,
    #     start_step=start_step
    # )

    frame = latents_to_image(main_latents)
    # full_image = get_image_from_frame(frame)
    # display(full_image)

    save_interval(index = loop_index, prompt = prompt, frame = frame, primer = primer_image)

    time.sleep(get_wait_time())


def get_rate_of_change(loop_index):
    sin = round((math.sin(loop_index) +1 ) /2, 4)
    cos = round((math.cos(loop_index) +1 ) /2, 4)
    return [sin, cos]

def get_primer_image():
    ret, frame = cam.read()
    if not ret: print("failed to grab frame")
    return frame


# TODO
def get_image_from_frame(frame):
    pass #upscale

# TODO
def display(frame):
    # cv2.imshow("test2", frame)
    pass

def save_interval(index, prompt, frame, primer):
    timestamp = datetime.datetime.now().isoformat()
    # cv2.imwrite(work_path + f"frame_{loop_index}_{prompt.replace(' ', '_')}.png", frame)
    cv2.imwrite(work_path + f"primer_{timestamp}}.png", primer)

# // to regulate FPS
def get_wait_time():
    # currently: # after previous loop finished and second/x passed.
    # future: # later may be controlled by performance monitor
    return 5000/1000



def run_main_loop():
    loop_index = 0 # TODO persist to file
    while True:
        loop_index+=1
        main_loop(loop_index)

# TODO
# create_new_latent_state(previous latents, input image, prompt, rate of change)
# to be converted into image & upscaled
# displayed on an output window

# save to files (input image, prompt, output image)




run_main_loop()

cam.release()
del(cam)
cv2.destroyAllWindows()

