import torch
from torch import autocast, float16
from torch.nn import functional as F

from lib.model import GPU

def prompt_to_embeds(prompt):
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

  # Cat for final embeddings
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
  return text_embeddings
