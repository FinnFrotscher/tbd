import torch
from torch import autocast, float16
from torch.nn import functional as F

from lib.compute import GPU


# ### prompts from the prompt generator function
# create 10 primary prompts and 6 art styles.
# let each artsyle display for 10 seconds.
# after each full rotation on art styles, move to the next primary prompt. primary prompts are core archetypes from tarot, religion, etc.

# // the story
# every 10 steps, move to next story beat is loaded
# after every loop through the whole set counts as a situation loop
situations = [
    "walking on the edge of a mountain",
    "standing on the edge of a mountain",
    "sitting on top of a mountain",
    "laying on the edge of a mountain",

    "sitting in front of a throne",

    "a wild animal",
    "playing a wild animal",
    "taming a wild animal",
    "playing an animal",
    "taming a friendly animal",
    "playing with a friendly animal",
    "riding a wild animal",
]

# // wetter
# per *situation loop
scenes = [
    "muslim temple, pleople praying, mekkah, ",
    "buddhist temple",
    "indian temple",
    "christian church, crosses, jesus, altar, priest",
    "pagan celtic druid",
    "spaces spaceship galaxy",
    "calm park with lake",
    "skyscrapeer town houses city appartment",
    "summery day",
    "wethery windy stormy at the ocean, grey sky and clowds, stormy water with big waves and white foam",
    "winter day, in the mountain, large cave, big trees, warm fire, deer standing outside in the snow",
    "small village spring day colza field dancing, spring festival, flowers, honey, small bees, ",
]

# //
# per *scenestyle loops
# colors, feelings, artists, styles
arts = [
    "surrellism dali",
    "surrellism escher",
    "cubism",
    "impressionism",
    "painting by davinci",
    "japanese drawing",
    "neon colors, cyberpunk scene"
]

class Storyteller:
    prompt = ""

    def __init__(self):
        self.prompt = ""

    def beat(self, storybeat = 0):
        sin = 0
        cos = 0

        # prompt = map_promp_state_to_prompt(prompt_state(sin, cos, index))
        situation_index = storybeat % len(situations)
        situation_full_cycles = int( ( storybeat - situation_index) / len(situations) )
        situation = situations[situation_index]

        scene_index =  situation_full_cycles % len(scenes)
        scene_full_cycles = int( ( situation_full_cycles - scene_index) / len(scenes) )
        scene = scenes[scene_index]

        art_index =  scene_full_cycles % len(arts)
        art_full_cycles = int( ( scene_full_cycles - art_index) / len(arts) )
        art = arts[art_index]

        self.prompt = f"{scene} . {situation} . {art}"
        return self.prompt

    def to_embedding(self, prompt = self.prompt):
        # Tokenize text and get embeddings
        text_input = GPU.tokenizer(
            prompt,
            padding='max_length',
            max_length=GPU.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            text_embeddings = GPU.text_encoder(text_input.input_ids.to(GPU.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = GPU.tokenizer(
            [''] * len(prompt), padding='max_length',
            max_length= GPU.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = GPU.text_encoder(uncond_input.input_ids.to(GPU.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

story = Storyteller()
