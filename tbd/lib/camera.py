import cv2, os
from os import path
# cam = cv2.VideoCapture(0)

cwd = path.join(os.getcwd())

# cam.release()
# del(cam)

# # ret, frame = cam.read()
# # if not ret: print("failed to grab frame")

class Camera:
    def grab_image(self, index):
        # get file
        filename = 'output_{0:04d}.png'.format(index)
        image_path = path.normpath(path.join(cwd, '..', f'tbd/input/dance/{filename}'))
        image = cv2.imread(image_path)#, cv2.COLOR_BGR2RGB)

        # crop on center
        shape = image.shape
        h = min(shape[0], shape[1])
        w = min(shape[0], shape[1])
        x = shape[1]/2 - w/2
        y = shape[0]/2 - h/2
        image = image[int(y):int(y+h), int(x):int(x+w)]

        # scale to model input
        image = cv2.resize(image, (512, 512))
        return image

