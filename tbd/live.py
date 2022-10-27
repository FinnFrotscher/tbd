import os, datetime, cv2, math, time
from os import path
import climage

from lib.compute import GPU
from lib.camera import Camera
from lib.story import Storyteller
from lib.latent_image import LatentImage

from lib.state import save_loop

story = Storyteller()
latents = LatentImage()
camera = Camera()

# get latents from text prompt
# print latents as image

# get latents from camera
# print latents as image

# perturb and merge them
# print image


def main_loop(loop_index):
    story.beat(loop_index)
    print(story.prompt)

    latents.from_text(story.to_embedding(), num_steps = 50 )
    img = latents.to_image()
    print('img', img )
    display(img)

    # primer_image = camera.grab_image()
    # primer_latent_image = LatentImage().from_image(primer_image)
    # latents.perturb(scale = 0.02)
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


# TODO

def display(img):
    img_path = "/home/finn/code/tbd/tbd/output/tmp.png"
    img.save(img_path)
    # output = climage.convert(img_path, width=20, is_unicode=True)
    # print(output)
    # os.remove(img_path)

    # cv2.imshow("test2", frame)
    # cv2.namedWindow("test2")


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

def get_rate_of_change(loop_index):
    sin = round((math.sin(loop_index) +1 ) /2, 4)
    cos = round((math.cos(loop_index) +1 ) /2, 4)
    return [sin, cos]


# TODO
def get_image_from_frame(frame):
    pass #upscale


def save_interval(index, prompt, frame, primer):
    timestamp = datetime.datetime.now().isoformat()
    # cv2.imwrite(work_path + f"frame_{loop_index}_{prompt.replace(' ', '_')}.png", frame)
    cv2.imwrite(work_path + f"primer_{timestamp}.png", primer)


# // to regulate FPS
def get_wait_time():
    # currently: # after previous loop finished and second/x passed.
    # future: # later may be controlled by performance monitor
    return 5000/1000


# TODO
# create_new_latent_state(previous latents, input image, prompt, rate of change)
# to be converted into image & upscaled
# displayed on an output window
# save to files (input image, prompt, output image)




run_main_loop()

# cv2.destroyAllWindows()

