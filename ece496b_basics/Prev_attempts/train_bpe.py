

from collections import defaultdict
import os
from pydoc import text
import regex as re

class BPE_outs:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str])  -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Vocabulary Initialization
    if os.path.exists(input_path):
        print("exists")
        with open(input_path, "r", encoding="utf-8") as file:
            init_data = file.read()
    else:
        print("Error, File not found")
        init_data = """Byte Pair Encoding (BPE) is a simple data compression algorithm that iteratively merges 
        the most frequent pair of consecutive bytes in a text. It is widely used in subword tokenization 
        for natural language processing (NLP) tasks, such as training neural network language models."""
    
    # Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ptText = re.findall(PAT, init_data)
    byte_sequences = [token.encode("utf-8") for token in ptText]  # Convert tokens into byte sequences
    print("Initial byte sequences: ", byte_sequences)
    # Initialize vocab
    vocab: dict[int, bytes] = {s: bytes([s]) for s in range(256)}
    merges: list[tuple[int, int]] = []

    next_token_id = 256
    # BPE Training Loop
    for i in range(vocab_size - 256):
        # Count occurrences of byte pairs
        num_occur = defaultdict(int)
        for seq in byte_sequences:
            for j in range(len(seq) - 1):
                num_occur[(seq[j:j+1], seq[j+1:j+2])] += 1
        
        if not num_occur:  # Stop if no more merges can be done
            print("No more merges")
            break

        # Find most frequent byte pair
        freq_pair = max(num_occur, key=num_occur.get)
        b1,b2 = freq_pair
        new_token=bytes(b1 + b2)

        # Assign new token ID
        vocab[next_token_id] = new_token
        merges.append((b1, b2)) # Store merge

        # Merge occurrences in byte_sequences
        byte_sequences = merge(byte_sequences, freq_pair, next_token_id)


        print(f"Merge {freq_pair[0]} and {freq_pair[1]}, output: {new_token}")
        next_token_id += 1 
        print("Current byte sequences:", byte_sequences)

    return vocab, merges


def merge(sequences: list[bytes], pair: tuple[bytes, bytes], new_token: int) -> list[bytes]:
    new_sequences = []

    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            # If current byte and next byte form the pair, merge them
            if i < len(seq) - 1 and seq[i:i+1] == pair[0] and seq[i+1:i+2] == pair[1]:
                #new_seq.append(new_token)  # Append the new merged token
                i += 2  # Skip the next character since we've merged this pair
                new_sequences.append(b"".join(new_seq))
                new_sequences.append(new_token)
                new_seq = []
            else:
                new_seq.append(seq[i:i+1])  # Otherwise, append the current character
                i += 1
        new_sequences.append(b"".join(new_seq))  # Join the new sequence together

    return new_sequences







if __name__ == "__main__":

    input_path = "test"
    vocab_size = 260
    special_tokens = ["<PAD>", "<UNK>"]

    result = train_bpe(input_path, vocab_size, special_tokens)

    

#"But the huge bonus prize is the real draw -- announced by an electronic display that resembles the ticking wheel on the TV game show , placed just above eye level . As her losses mounted to more than $200 , Budz fed the machine $5 tokens , pressing the Spin button almost rhythmically -- no serious slot player touches the pull handle on a one-armed bandit"