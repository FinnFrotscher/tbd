import torch
from torch import autocast, float16
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer

device = 'cuda'

# 2. Load the tokenizer and text encoder to tokenize and encode the text

tokenizer = CLIPTokenizer.from_pretrained('/home/finn/data/stable-diffusion-v1-4/tokenizer')
text_encoder = CLIPTextModel.from_pretrained('/home/finn/data/stable-diffusion-v1-4/text_encoder')
text_encoder.to(device)


def prompt_to_embeds(prompt):
  # Tokenize text and get embeddings
  text_input = tokenizer(
      prompt,
      padding='max_length',
      max_length=tokenizer.model_max_length,
      truncation=True,
      return_tensors='pt'
    )

  with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

  # Do the same for unconditional embeddings
  uncond_input = tokenizer(
      [''] * len(prompt), padding='max_length',
      max_length=tokenizer.model_max_length, return_tensors='pt')
  with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

  # Cat for final embeddings
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
  return text_embeddings
