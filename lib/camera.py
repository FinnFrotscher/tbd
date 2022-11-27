import cv2, os
import numpy as np
from lib.image import Image
from globals import *
# cam = cv2.VideoCapture(0)

cwd = os.path.join(os.getcwd())

# cam.release()
# del(cam)

# # ret, frame = cam.read()
# # if not ret: print("failed to grab frame")

size = output_dimensions

class Camera:
    def grab_image(self, index):
        filename = 'output_{0:04d}.png'.format(index)
        image_path = os.path.normpath(os.path.join(cwd, '..', f'input/dance/{filename}'))

        image = Image(from_path = image_path)
        image.trim_to_square()
        image.resize(size)
        return image

