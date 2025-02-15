
from dataclasses import dataclass
from collections import defaultdict
import os
import regex as re
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: Dict[int, bytes]             # index -> bytes
    merges: Dict[Tuple[int, int], int] 


def pre_tokenize(text: str)-> List[int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = re.findall(PAT, text) #Pretokenize using above regex
    indices = [b for token in tokens for b in token.encode("utf-8")] #Flatten into byte strings using utf-8
    return indices


def train_bpe(vocab_size: int, input_path: str | os.PathLike, special_tokens: list[str]) -> BPETokenizerParams:
    if os.path.exists(input_path):
        print("exists")
        with open(input_path, "r", encoding="utf-8") as file:
            init_data = file.read()
    else:
        print("Error, File not found")
        init_data = """Byte Pair Encoding (BPE) is a simple data compression algorithm that iteratively merges 
        the most frequent pair of consecutive bytes in a text. It is widely used in subword tokenization 
        for natural language processing (NLP) tasks, such as training neural network language models."""
    #Pre tokenize the text and flatten it into byte sequences
    indices = pre_tokenize(init_data)

    # index1, index2 => merged index
    merges: Dict[Tuple[int, int], int] = {}

    # index -> bytes
    vocab: Dict[int, bytes] = {
        x: bytes([x]) for x in range(256)
    }

    for j in range(len(special_tokens)):
        vocab[256+j] = special_tokens[j]

    init_vocab_size = 256 + len(special_tokens)

    num_merges = vocab_size - 256
    if num_merges <= 0: return vocab, merges


    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for pair in zip(indices, indices[1:]):  # For each adjacent pair
            counts[pair] += 1

        # Find the most common pair.
        pair = max(counts, key=counts.get)

        # Merge that pair.
        new_index = init_vocab_size + i
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

        print(f"Merge {vocab[pair[0]]} {vocab[pair[1]]} -> {vocab[new_index]}")
        indices = merge(indices, pair, new_index)

        print(f"Text: {list(map(vocab.get, indices))}")

    return BPETokenizerParams(vocab=vocab, merges=merges)


def merge(indices: List[int], pair: Tuple[int, int], new_index: int) -> List[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


if __name__ == "__main__":

    input_path = "test"
    vocab_size = 257
    special_tokens = ["<PAD>", "<UNK>"]

    result = train_bpe(300,"temp", special_tokens)
    print(result)