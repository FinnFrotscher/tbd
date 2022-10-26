import cv2, os
from os import path
# cam = cv2.VideoCapture(0)

cwd = path.join(os.getcwd())
input_path = 'tbd/input/street500.jpg'

image_path = path.normpath(path.join(cwd, '..', input_path))

def grab_image():
    # ret, frame = cam.read()
    primer = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    # if not ret: print("failed to grab frame")
    return primer
