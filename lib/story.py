import torch
from torch import autocast, float16
from torch.nn import functional as F

from lib.prompt.storyblocks import situations, scenes, arts
from lib.compute import GPU


class Storyteller:
    prompt = ""

    def __init__(self, storybeat = 0):
        # self.prompt = self.beat(storybeat = storybeat)
        self.prompt = "people dancing around the fire at night, under a starlit sky showing the milky way, painted in a romanticism style, lots red and blue, highly detailed, 4k "

    def beat(self, storybeat = 0):
        return
        situation_index = storybeat % len(situations)
        situation_full_cycles = int( ( storybeat - situation_index) / len(situations) )
        situation = situations[situation_index]

        scene_index =  situation_full_cycles % len(scenes)
        scene_full_cycles = int(( situation_full_cycles - scene_index) / len(scenes))
        scene = scenes[scene_index]

        art_index =  scene_full_cycles % len(arts)
        art_full_cycles = int( ( scene_full_cycles - art_index) / len(arts) )
        art = arts[art_index]

        self.prompt = f"{scene}, {situation}, {art}"
        return self.prompt

    def to_embedding(self):
        # print('prompt',self.prompt)
        # Tokenize text and get embeddings
        prompt = [ self.prompt ]
        text_input = GPU.tokenizer(prompt, padding='max_length', max_length=GPU.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = GPU.text_encoder(text_input.input_ids.to(GPU.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = GPU.tokenizer(
            [''] * len(prompt), padding='max_length',
            max_length=GPU.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = GPU.text_encoder(uncond_input.input_ids.to(GPU.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

