import json
import numpy as np
import regex as re
from typing import Dict, Iterable, Iterator, List, Tuple, Optional


class BPETokenizer:

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        
        self.vocab = vocab 
        self.merges = merges
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []

        #Add special tokens to vocab if they aren't already there
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Creates a BPETokenizer from serialized vocab and merges files."""
        
        # Load vocabulary from a JSON file
        with open(vocab_filepath, "r", encoding="utf-8") as vocab_file:
            vocab_data = json.load(vocab_file)

        # Convert loaded vocab to expected format (id -> bytes)
        vocab = {int(k): bytes(v, encoding="utf-8") for k, v in vocab_data.items()}

        # Load merges from a text file
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as merges_file:
            for line in merges_file:
                pair = line.strip().split()
                if len(pair) == 2:
                    merges.append((pair[0].encode("utf-8"), pair[1].encode("utf-8")))

        # Instantiate and return BPETokenizer
        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, tokens: List[int]) -> List[bytes]:
        # Convert integers (ASCII values) back to bytes
        tokens = [bytes([t]) for t in tokens]

        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            merge_candidates = {pair: i for i, pair in enumerate(pairs) if pair in self.merges}

            if not merge_candidates:
                return tokens  # When we run out of tokens to merge, return result

            # Find best pair (first occurrence)
            best_pair = min(merge_candidates, key=lambda p: self.merges.index(p))

            # Merge tokens
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])  # Merge as bytes
                    i += 2  # Skip next token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens


    def encode(self, text: str) -> List[int]:
        """Takes input text and returns a list of token IDs, ensuring special tokens are preserved."""
        
        # Ensure special tokens are properly tokenized and separated
        for token in self.special_tokens:
            if token in text:
                text = text.replace(token, f" {token} ")  # Ensure separation
        
        # Pre-tokenize using GPT-2 pattern
        pre_tokenized = re.findall(r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+", text)

        # Encode in utf-8
        byte_tokens = [token.encode("utf-8") for token in pre_tokenized]

        # Apply BPE merges
        merged_tokens = []
        for token in byte_tokens:
            bpe_tokens = self._bpe_merge(list(token))
            merged_tokens.extend(bpe_tokens)

        # Convert to token IDs, ensuring special tokens remain in the final output
        token_ids = [self.token_to_id[token] for token in merged_tokens if token in self.token_to_id]

        return token_ids


    def decode(self, token_ids: List[int]) -> str:
        """Takes a list of token IDs and returns a string of text, ensuring special tokens remain intact."""
        
        byte_string = b"".join(self.vocab[t] for t in token_ids)
        text = byte_string.decode("utf-8", errors="replace")  # Use "replace" to avoid crashes

        # Restore special tokens (if they exist in text)
        for token in self.special_tokens:
            text = text.replace(f" {token} ", token)
        return text

    def add_special_token(self, token: str):
        #Handle sepcial tokens
        token_bytes = token.encode("utf-8")
        if token_bytes not in self.token_to_id:
            new_id = len(self.vocab)
            self.vocab[new_id] = token_bytes
            self.token_to_id[token_bytes] = new_id
            self.special_tokens.append(token)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Tokenizes an iterable lazily, yielding token IDs to save memory."""
        for line in iterable:
            yield from self.encode(line)


if __name__ == "__main__":
    # Example vocab and merges
    vocab = {
        0: b'\x00', 1: b'\x01', 2: b'\x02', 3: b'\x03', 4: b'\x04', 5: b'\x05',
        6: b'\x06', 7: b'\x07', 8: b'\x08', 9: b'\t', 10: b'\n', 32: b' ', 97: b'a',
        98: b'b', 99: b'c', 100: b'd', 101: b'e', 102: b'f', 103: b'g', 104: b'h',
        105: b'i', 106: b'j', 107: b'k', 108: b'l', 109: b'm', 110: b'n', 111: b'o',
        112: b'p', 113: b'q', 114: b'r', 115: b's', 116: b't', 117: b'u', 118: b'v',
        119: b'w', 120: b'x', 121: b'y', 122: b'z', 256: b'th', 257: b' th', 258: b' thi'
    }
    merges = [(b't', b'h'), (b' ', b'th'), (b' th', b'i')]

    # Initialize Tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    #Load TinyStories sample text
    with open("tests/fixtures/tinystories_sample.txt", "r", encoding="utf-8") as f:
        tinystories_text = f.read()

    #Tokenize the dataset
    tokenized_data = tokenizer.encode(tinystories_text)

    #Save tokenized output as a NumPy file
    np.save("tinystories_tokens.npy", np.array(tokenized_data, dtype=np.int32))

    print("Tokenized dataset saved as tinystories_tokens.npy")