import regex as re
from typing import Dict, List, Tuple, Optional


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

    def _bpe_merge(self, tokens: List[int]) -> List[bytes]:
        #Apply bpe merges iterativly

        #Convert integers (ASCII values) back to bytes
        tokens = [bytes([t]) for t in tokens]

        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            merge_candidates = {pair: i for i, pair in enumerate(pairs) if pair in self.merges}
            
            if not merge_candidates:
                #When we run out of tokens to merge
                return tokens 
            
            #Find best pair
            best_pair = min(merge_candidates, key=lambda p: self.merges.index(p))

            #Merge tokens
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])  #Merge as bytes
                    i += 2  #Skip next token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        #Takes input text and returns a list of IDs

        #Handle special tokens
        for token in self.special_tokens:
            if token in text:
                text = text.replace(token, f" {token} ")  #Ensure separation

        #Pre-tokenize using GPT-2 pattern
        pre_tokenized = re.findall(r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+", text)

        #Encode in utf-8
        byte_tokens = [token.encode("utf-8") for token in pre_tokenized]

        #Apply BPE merges
        merged_tokens = []
        for token in byte_tokens:
            bpe_tokens = self._bpe_merge(list(token))
            merged_tokens.extend(bpe_tokens)

        #Convert to token IDs
        token_ids = [self.token_to_id[token] for token in merged_tokens if token in self.token_to_id]

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        #Takes a list of token IDs and returns a string of text

        byte_string = b"".join(self.vocab[t] for t in token_ids)
        return byte_string.decode("utf-8", errors="replace")  #Use "replace" to avoid invalid sequences

    def add_special_token(self, token: str):
        #Handle sepcial tokens
        token_bytes = token.encode("utf-8")
        if token_bytes not in self.token_to_id:
            new_id = len(self.vocab)
            self.vocab[new_id] = token_bytes
            self.token_to_id[token_bytes] = new_id
            self.special_tokens.append(token)



if __name__ == "__main__":
    vocab = {
        0: b'\x00', 1: b'\x01', 2: b'\x02', 3: b'\x03', 4: b'\x04', 5: b'\x05', 6: b'\x06', 7: b'\x07', 8: b'\x08', 
        9: b'\t', 10: b'\n', 11: b'\x0b', 12: b'\x0c', 13: b'\r', 14: b'\x0e', 15: b'\x0f', 16: b'\x10', 17: b'\x11', 
        18: b'\x12', 19: b'\x13', 20: b'\x14', 21: b'\x15', 22: b'\x16', 23: b'\x17', 24: b'\x18', 25: b'\x19', 26: b'\x1a', 
        27: b'\x1b', 28: b'\x1c', 29: b'\x1d', 30: b'\x1e', 31: b'\x1f', 32: b' ', 33: b'!', 34: b'"', 35: b'#', 36: b'$', 
        37: b'%', 38: b'&', 39: b"'", 40: b'(', 41: b')', 42: b'*', 43: b'+', 44: b',', 45: b'-', 46: b'.', 47: b'/', 48: b'0', 
        49: b'1', 50: b'2', 51: b'3', 52: b'4', 53: b'5', 54: b'6', 55: b'7', 56: b'8', 57: b'9', 58: b':', 59: b';', 60: b'<', 
        61: b'=', 62: b'>', 63: b'?', 64: b'@', 65: b'A', 66: b'B', 67: b'C', 68: b'D', 69: b'E', 70: b'F', 71: b'G', 72: b'H', 
        73: b'I', 74: b'J', 75: b'K', 76: b'L', 77: b'M', 78: b'N', 79: b'O', 80: b'P', 81: b'Q', 82: b'R', 83: b'S', 84: b'T', 
        85: b'U', 86: b'V', 87: b'W', 88: b'X', 89: b'Y', 90: b'Z', 91: b'[', 92: b'\\', 93: b']', 94: b'^', 95: b'_', 96: b'`', 
        97: b'a', 98: b'b', 99: b'c', 100: b'd', 101: b'e', 102: b'f', 103: b'g', 104: b'h', 105: b'i', 106: b'j', 107: b'k', 
        108: b'l', 109: b'm', 110: b'n', 111: b'o', 112: b'p', 113: b'q', 114: b'r', 115: b's', 116: b't', 117: b'u', 118: b'v', 
        119: b'w', 120: b'x', 121: b'y', 122: b'z', 123: b'{', 124: b'|', 125: b'}', 126: b'~', 127: b'\x7f', 128: b'\x80', 
        129: b'\x81', 130: b'\x82', 131: b'\x83', 132: b'\x84', 133: b'\x85', 134: b'\x86', 135: b'\x87', 136: b'\x88', 137: b'\x89',
          138: b'\x8a', 139: b'\x8b', 140: b'\x8c', 141: b'\x8d', 142: b'\x8e', 143: b'\x8f', 144: b'\x90', 145: b'\x91', 
          146: b'\x92', 147: b'\x93', 148: b'\x94', 149: b'\x95', 150: b'\x96', 151: b'\x97', 152: b'\x98', 153: b'\x99', 
          154: b'\x9a', 155: b'\x9b', 156: b'\x9c', 157: b'\x9d', 158: b'\x9e', 159: b'\x9f', 160: b'\xa0', 161: b'\xa1', 
          162: b'\xa2', 163: b'\xa3', 164: b'\xa4', 165: b'\xa5', 166: b'\xa6', 167: b'\xa7', 168: b'\xa8', 169: b'\xa9', 
          170: b'\xaa', 171: b'\xab', 172: b'\xac', 173: b'\xad', 174: b'\xae', 175: b'\xaf', 176: b'\xb0', 177: b'\xb1', 
          178: b'\xb2', 179: b'\xb3', 180: b'\xb4', 181: b'\xb5', 182: b'\xb6', 183: b'\xb7', 184: b'\xb8', 185: b'\xb9', 
          186: b'\xba', 187: b'\xbb', 188: b'\xbc', 189: b'\xbd', 190: b'\xbe', 191: b'\xbf', 192: b'\xc0', 193: b'\xc1', 
          194: b'\xc2', 195: b'\xc3', 196: b'\xc4', 197: b'\xc5', 198: b'\xc6', 199: b'\xc7', 200: b'\xc8', 201: b'\xc9', 
          202: b'\xca', 203: b'\xcb', 204: b'\xcc', 205: b'\xcd', 206: b'\xce', 207: b'\xcf', 208: b'\xd0', 209: b'\xd1', 
          210: b'\xd2', 211: b'\xd3', 212: b'\xd4', 213: b'\xd5', 214: b'\xd6', 215: b'\xd7', 216: b'\xd8', 217: b'\xd9', 
          218: b'\xda', 219: b'\xdb', 220: b'\xdc', 221: b'\xdd', 222: b'\xde', 223: b'\xdf', 224: b'\xe0', 225: b'\xe1', 
          226: b'\xe2', 227: b'\xe3', 228: b'\xe4', 229: b'\xe5', 230: b'\xe6', 231: b'\xe7', 232: b'\xe8', 233: b'\xe9', 
          234: b'\xea', 235: b'\xeb', 236: b'\xec', 237: b'\xed', 238: b'\xee', 239: b'\xef', 240: b'\xf0', 241: b'\xf1', 
          242: b'\xf2', 243: b'\xf3', 244: b'\xf4', 245: b'\xf5', 246: b'\xf6', 247: b'\xf7', 248: b'\xf8', 249: b'\xf9', 
          250: b'\xfa', 251: b'\xfb', 252: b'\xfc', 253: b'\xfd', 254: b'\xfe', 255: b'\xff',
        256: b'th', 257: b' th', 258: b' thi', 259: b' this', 260: b' the', 261: b' tha', 262: b' that', 263: b'the'
    }
    merges = [(b't', b'h'), (b' ', b'th'), (b' th', b'i'), (b' thi', b's'), (b' th', b'e'), (b' th', b'a'), (b' tha', b't'), (b'th', b'e')]

    #Initialize Tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    #Encoding text
    encoded = tokenizer.encode("this is the test")
    print("Encoded:", encoded) 

    #Decoding
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    #Adding a special token
    tokenizer.add_special_token("<NEW_TOKEN>")
    print("New Token ID:", tokenizer.token_to_id[b"<NEW_TOKEN>"])  #Should be added to vocab