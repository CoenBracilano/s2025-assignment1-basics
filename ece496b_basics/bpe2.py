import regex as re
import collections
from typing import Tuple, List, Dict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(text: str) -> List[List[int]]:
    """
    Pre-tokenizes input text using GPT-2 regex pattern and converts it to byte-level token IDs.
    """
    tokens = re.findall(GPT2_PAT, text)
    return [[b for b in token.encode("utf-8")] for token in tokens]  #Convert to byte token IDs

def get_vocab(corpus: List[List[int]]) -> Dict[Tuple[int, int], int]:
    #Find pairs of characters
    vocab = collections.defaultdict(int)
    for word in corpus:
        for i in range(len(word) - 1):
            vocab[(word[i], word[i + 1])] += 1  #Store pair counts
    return vocab

def merge_vocab(pair: Tuple[int, int], corpus: List[List[int]], new_token_id: int) -> List[List[int]]:

    new_corpus = []
    for word in corpus:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_token_id)  #Replace with new token ID
                i += 2  #Skip next token
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    #Pre-tokenize text using GPT-2 pattern
    corpus = pre_tokenize(text)

    #Initialize vocabulary 
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    while len(vocab) < vocab_size:
        print("Run number ",(len(vocab) - vocab_size),"of ", (vocab_size-256))
        pairs = get_vocab(corpus)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)

        #Create a new token ID for the merged token
        new_token_id = len(vocab)
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]]  #Merge byte sequences
        vocab[new_token_id] = new_token  #Store as bytes

        #Store merges using byte sequences
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        #Update corpus: replace token pairs with the new token ID
        corpus = merge_vocab(best_pair, corpus, new_token_id)

    #Add special tokens at the end
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()

    return vocab, merges


if __name__ == "__main__":
    input_file = "data\TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]

    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
    
    print("Vocabulary:", vocab)
    print("Merges:", merges)