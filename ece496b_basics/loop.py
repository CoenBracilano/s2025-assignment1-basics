import os
from typing import IO, BinaryIO
import torch
import torch.nn as nn
import math
import numpy.typing as npt
import numpy as np
from torch.optim.optimizer import Optimizer


# def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
#     x = npt.array(dataset, dtype=npt.int64)

#     num_tokens = len(dataset)
#     start_indices = npt.random.randint(0, num_tokens - context_length, size=batch_size)
#     input_sequences = []
#     next_token_targets = []

#     for i in start_indices:
#         input_seq = x[i:i + context_length]
#         target_seq = x[i + 1:i + context_length + 1]
        
#         input_sequences.append(input_seq)
#         next_token_targets.append(target_seq)

#     input_sequences = torch.tensor(input_sequences, device=device)
#     next_token_targets = torch.tensor(next_token_targets, device=device)
    
#     return input_sequences, next_token_targets

def data_loader(dataset: np.ndarray, batch_size: int, context_length: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length - 1
    if max_start_idx <= 0:
        raise ValueError("Dataset is too short for the given context_length.")
    
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    inputs = np.array([dataset[i : i + context_length] for i in start_indices], dtype=np.int64)
    labels = np.array([dataset[i + 1 : i + context_length + 1] for i in start_indices], dtype=np.int64)
    
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return inputs_tensor, labels_tensor


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out)


def load_checkpoint(src:str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer)-> int:
   checkpoint = torch.load(src)
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   return checkpoint['iteration']



