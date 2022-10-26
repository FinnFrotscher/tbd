# main loop function

## inputs
### prompts from the prompt generator function
create 10 primary prompts and 6 art styles.
let each artsyle display for 10 seconds.
after each full rotation on art styles, move to the next primary prompt. primary prompts are core archetypes from tarot, religion, etc.

### image from the video feed
can be just a simple image in the beginning or later on a image, a blur map, and an object map.
the object map can be used to lighten or strengthen the effect in selected areas of the map.
the blur map can be used to soften corners and to create focal points for detail.
those could be based on a image processing pipeline that works out contrast or from an ML model that detects objects, spaces, foreground and background etc.
during development use video frames.

### latent state from the previous image
change the image only slightly between each training round.

### create new latent state
latent space conditionin, number of iterations = rate of change


## procedure
### input
rate of change

prompt = previous prompt state * rate of change

create new latent state (previous patents, input image, prompt, rate of change)

to be converted into image
upscale the image.
save to files (input image, prompt, output image)
displayed on an output window


run main loop
after previous loop finished and second/x passed.
