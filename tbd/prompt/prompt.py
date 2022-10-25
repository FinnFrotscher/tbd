

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

def get_prompt(sin : 0, cos = 0, index = 0):
    # prompt = map_promp_state_to_prompt(prompt_state(sin, cos, index))
    situation_index = index % len(situations)
    situation_full_cycles = int( ( index - situation_index) / len(situations) )
    situation = situations[situation_index]

    scene_index =  situation_full_cycles % len(scenes)
    scene_full_cycles = int( ( situation_full_cycles - scene_index) / len(scenes) )
    scene = scenes[scene_index]

    art_index =  scene_full_cycles % len(arts)
    art_full_cycles = int( ( scene_full_cycles - art_index) / len(arts) )
    art = arts[art_index]

    prompt = f"{scene} . {situation} . {art}"
    return prompt


# // input    dimensional state
# dimensional state is n object of an object that can be represented in more then two numberic dimensions
# it must follow certain rules, like for a theoretic 2d object *any change in x must be equated with a change in y that would be deducable if the two dimensions xy where to follow the trajectory of a circle.*
# // the two dimensions we receive are x=*rate of change of new_latent_number_of_steps, and y=old_latent_fuzzy_multiplie*
