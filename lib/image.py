import cv2, os
from pathlib import Path
from PIL import Image as PILImage, ImageDraw as PILImageDraw
import numpy as np

cwd = os.path.join(os.getcwd())

class Image:
    raw = None
    def __init__(self, img=None, from_path=None, pil_img=None):
        if(pil_img is not None):
            self.raw = numpy.array(pil_image)
        elif(from_path is not None):
            self.raw = cv2.imread(from_path)
        else:
            self.raw = img

    def store(self, filename):
        input_path = f'output/{filename}'
        image_path = os.path.normpath(os.path.join(cwd, '..', input_path))
        Path(os.path.normpath(os.path.join(image_path, '..'))).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_path, self.raw)

    def to_PIL(self):
        image = cv2.cvtColor(self.raw, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(image)

    def trim_to_square(self):
        # crop on center
        shape = self.raw.shape
        h = min(shape[0], shape[1])
        w = min(shape[0], shape[1])
        x = shape[1]/2 - w/2
        y = shape[0]/2 - h/2
        self.raw = self.raw[int(y):int(y+h), int(x):int(x+w)]

    def resize(self, s):
        self.raw = cv2.resize(self.raw, (s, s))

    def invert(self):
        self.raw = (255 - self.raw)

    def gray(self):
        self.raw = cv2.cvtColor(self.raw, cv2.COLOR_BGR2GRAY)
        self.raw = cv2.cvtColor(self.raw, cv2.COLOR_GRAY2BGR)

    def apply_brightness_contrast(self, brightness = 80, contrast = 80):
        # out[row:row+s, col:col+s] = apply_brightness_contrast(img, b, c)
        buf = self.raw.copy()

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow

            buf = cv2.addWeighted(buf, alpha_b, buf, 0, gamma_b)

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        self.raw = buf



# Open a typical 24 bit color image. For this kind of image there are
# 8 bits (0 to 255) per color channel
# cam = Camera()
# img =  cam.grab_image(105) # mandrill reference image from USC SIPI

# font = cv2.FONT_HERSHEY_SIMPLEX
# fcolor = (0,0,0)

# blist = [127, 135, 144, 127, 135, 144] # list of brightness values
# clist = [64, 64, 64, 70, 70, 70] # list of contrast values
# out = np.zeros((s*2, s*3, 3), dtype = np.uint8)

# # could be made into a function that iterates on combinations of brightness and contrast until it finds 
# # a pair of values that result in an average of Y > np.average( out ) > X
# # by recursively calling itself and adjusting c,b 

# for i, b in enumerate(blist):
#     c = clist[i]
#     print(f'brightness:{b}, contrast:{c}')
#     row = s*int(i/3)
#     col = s*(i%3)
    
#     out[row:row+s, col:col+s] = apply_brightness_contrast(img, b, c)
#     msg = 'b %d' % b
#     cv2.putText(out,msg,(col,row+s-22), font, .7, fcolor,1,cv2.LINE_AA)
#     msg = 'c %d' % c
#     cv2.putText(out,msg,(col,row+s-4), font, .7, fcolor,1,cv2.LINE_AA)
#     print(np.average( out[row:row+s, col:col+s]) )

# image_path = path.normpath(path.join(cwd, '..', 'tbd/output/output.png'))
# cv2.imwrite(image_path, out)


# https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
