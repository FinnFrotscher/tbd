
# ### prompts from the prompt generator function
# create 10 primary prompts and 6 art styles.
# let each artsyle display for 10 seconds.
# after each full rotation on art styles, move to the next primary prompt. primary prompts are core archetypes from tarot, religion, etc.

# // the story
# every 10 steps, move to next story beat is loaded
# after every loop through the whole set counts as a situation loop
situations = [
    "sitting on a throne",

    "walking on the edge of a mountain",

    "standing on the edge of a mountain",
    "sitting on top of a mountain",
    "laying on the edge of a mountain",


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
    "wizard",
    "summery day",
    "muslim temple, pleople praying, mekkah, ",
    "buddhist temple",
    "indian temple",
    "christian church, crosses, jesus, altar, priest",
    "spaces spaceship galaxy",
    "calm park with lake",
    "skyscrapeer town houses city appartment",
    "wethery windy stormy at the ocean, grey sky and clowds, stormy water with big waves and white foam",
    "winter day, in the mountain, large cave, big trees, warm fire, deer standing outside in the snow",
    "small village spring day colza field dancing, spring festival, flowers, honey, small bees, ",
]

# //
# per *scenestyle loops
# colors, feelings, artists, styles
arts = [
    "photorealistic",
    "surrellism dali",
    "surrellism escher",
    "cubism",
    "impressionism",
    "painting by davinci",
    "japanese drawing",
    "neon colors, cyberpunk scene"
]
