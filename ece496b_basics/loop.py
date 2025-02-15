import os
from typing import IO, BinaryIO
import torch
import numpy as np

from ece496b_basics.gelu import softmax


#Batch and load a data set
def data_loader(dataset: np.ndarray, batch_size: int, context_length: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length - 1
    if max_start_idx <= 0:
        raise ValueError("Dataset is too short for the given context_length.", len(dataset))
    
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    inputs = np.array([dataset[i : i + context_length] for i in start_indices], dtype=np.int64)
    labels = np.array([dataset[i + 1 : i + context_length + 1] for i in start_indices], dtype=np.int64)
    
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return inputs_tensor, labels_tensor

#Save a checkpoint to a file given a model
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out)

#load a checkpoint from for a model
def load_checkpoint(src:str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer)-> int:
   checkpoint = torch.load(src)
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   return checkpoint['iteration']


def decode(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_p=0.9, device="cpu"):

     model.eval()

     input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
     for _ in range(max_tokens):
        with torch.no_grad():
            # Forward pass through the model
            logits = model(input_ids)  # Shape: [1, seq_len, vocab_size]
            
            # Get last token logits (next-word prediction)
            logits = logits[:, -1, :]  # Shape: [1, vocab_size]
            
            # Apply softmax temperature scaling
            logits = logits / temperature
            probs = softmax(logits, dim=-1)

            # Top-p nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Select tokens up to top_p threshold
            top_p_mask = cumulative_probs < top_p
            top_p_mask = torch.cat((torch.ones_like(top_p_mask[:, :1]), top_p_mask[:, :-1]), dim=-1)

            filtered_probs = sorted_probs * top_p_mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)  # Normalize

            # Sample next token
            next_token = sorted_indices[torch.multinomial(filtered_probs, 1)]

            # Append new token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            # If we generate <|endoftext|>, stop
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode tokens back to text
        generated_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
        
        return generated_text



