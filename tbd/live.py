import os, datetime, cv2, math, time
from os import path

from lib.compute import GPU
from lib.state import save_loop
from lib.camera import grab_image

from lib.prompt import Prompt
from lib.latent_image import latents


# cv2.namedWindow("test2")

# latents = load_first_latents()

def main_loop(loop_index):
    # [sin, cos] = get_rate_of_change(loop_index)
    prompt = get_prompt(loop_index)
    text_embeds = prompt_to_embeds(prompt)

    primer_image = get_primer_image()
    primer_image_latents = image_to_latents(primer_image)

    latents.perturb(scale = 0.02)

    # weigh this operation by rate_of_change.cos
    main_latents = main_latents * primer_image_latents

    latents = produce_latents(
        text_embeds = text_embeds,
        latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        height=height, width=width,
        start_step=start_step
    )

    # save_loop(index = loop_index, prompt = prompt, primer = primer_image, image = None )
    # TODO
    # i have the current latents, the primer latents, the prompt embeddings
    # and a sin and cosin wave
    # i want out the latent space that was mutated by prompt embeddings

    # i can also blurr any latents  (by pertubing by scale X)
    # and deblurs towards the prompt (by num_inference_steps Y)

    # i will combine primer and current latents by matric multiplication
    # if that doesnt work i will have to merge the final image from the previous loop with the actual image from the camera with an aplpha effect and extract the latents of that.
    # i will blurr the combined latent space of the primer and the current latents


    #x frame = latents_to_image(main_latents)
    # full_image = get_image_from_frame(frame)
    # display(full_image)

    #x save_interval(index = loop_index, prompt = prompt, frame = frame, primer = primer_image)

    #x time.sleep(get_wait_time())


def get_rate_of_change(loop_index):
    sin = round((math.sin(loop_index) +1 ) /2, 4)
    cos = round((math.cos(loop_index) +1 ) /2, 4)
    return [sin, cos]



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
    cv2.imwrite(work_path + f"primer_{timestamp}.png", primer)

# // to regulate FPS
def get_wait_time():
    # currently: # after previous loop finished and second/x passed.
    # future: # later may be controlled by performance monitor
    return 5000/1000



def run_main_loop():
    loop_index = 0 # TODO persist to file
    try:
        main_loop(loop_index)
        while False:
            loop_index+=1
            main_loop(loop_index)
    finally:
        print('Clean')
        GPU.clean()

# TODO
# create_new_latent_state(previous latents, input image, prompt, rate of change)
# to be converted into image & upscaled
# displayed on an output window

# save to files (input image, prompt, output image)




run_main_loop()

# cam.release()
# del(cam)
# cv2.destroyAllWindows()

