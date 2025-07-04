from itertools import pairwise
import multiprocessing
import pickle
from typing import Iterator
import numpy as np
import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.crlf2lf import crlf2lf

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[bytes, int]:
    '''
    input_path: str Path to a text file with BPE tokenizer training data.
    start: int Start byte offset for the chunk to be pre-tokenized.
    end: int End byte offset for the chunk to be pre-tokenized.
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.

    Your pre-tokenization function should return a dictionary mapping pre-tokens (bytes) to their counts.
    '''
    pat_special_token = "|".join(re.escape(s) for s in special_tokens)
    # Read the chunk from the file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks = re.split(pat_special_token, chunk)
    
    cnt_pretoken = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            cnt_pretoken[pre_token] = cnt_pretoken.get(pre_token, 0) + 1

    return cnt_pretoken

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.

    Your BPE training function should return the resulting vocabulary and merges:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
    lary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    '''
    # create the initial vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    # add special tokens to the vocabulary
    for i, token in enumerate(special_tokens):
        token_bytes = token.encode("utf-8")
        vocab[256+i] = token_bytes

    # pre-tokenize the input file
    # crlf2lf(input_path)
    num_processes = multiprocessing.cpu_count()
    input_args = []
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes*4, "<|endoftext|>".encode("utf-8"))
        for start, end in pairwise(chunk_boundaries):
            input_args.append((input_path, start, end, special_tokens))
        
    with multiprocessing.Pool(num_processes) as pool:
        cnts_pretoken = pool.starmap(pre_tokenize, input_args)
    
    # combine the counts from all processes
    cnt_pretoken_bytes = {}
    for cnt in cnts_pretoken:
        for pre_token, count in cnt.items():
            pretoken_bytes = pre_token.encode("utf-8")
            cnt_pretoken_bytes[pretoken_bytes] = cnt_pretoken_bytes.get(pretoken_bytes, 0)+count
    del cnts_pretoken

    # count pairs
    cnt_pairs = {}
    idx2pretoken_bytes = [None]*len(cnt_pretoken_bytes)
    token_length = [None]*len(cnt_pretoken_bytes) # store the length of each token in pretoken_bytes
    pos_pair = {}
    for i, (pretoken_bytes, count) in enumerate(cnt_pretoken_bytes.items()):
        idx2pretoken_bytes[i] = pretoken_bytes
        token_length[i] = np.ones(len(pretoken_bytes), dtype=np.uint16)
        for j in range(len(pretoken_bytes)-1):
            pair = (pretoken_bytes[j:j+1], pretoken_bytes[j+1:j+2])
            cnt_pairs[pair] = cnt_pairs.get(pair, 0)+count
            if pair not in pos_pair:
                pos_pair[pair] = []
            pos_pair[pair].append((i, j)) # pretoken index, pair index
    
    # merge pairs until vocab size is reached
    merges = []
    vocab_idx = len(vocab)
    num_merges = vocab_size-len(vocab)
    for _ in tqdm(range(num_merges)):
        # find the most common pair
        most_common_pair = max(cnt_pairs.keys(), key=lambda x: (cnt_pairs[x], x[0], x[1])) # use heap

        # merge the pair
        merges.append(most_common_pair)
        new_token = most_common_pair[0]+most_common_pair[1]
        vocab[vocab_idx] = new_token
        vocab_idx += 1

        # update the counts
        for pretoken_idx, pair_idx in pos_pair[most_common_pair]:
            pretoken_bytes = idx2pretoken_bytes[pretoken_idx]
            length = token_length[pretoken_idx]
            if length[pair_idx]!=len(most_common_pair[0]) or length[pair_idx+length[pair_idx]]!=len(most_common_pair[1]): # processed
                continue

            if pair_idx>0: # merge prev pair
                prev_idx_start = pair_idx-length[pair_idx-1]
                prev_token = pretoken_bytes[prev_idx_start:pair_idx]
                prev_pair = (prev_token, most_common_pair[0])
                new_prev_pair = (prev_token, new_token)
                cnt_pairs[prev_pair] -= cnt_pretoken_bytes[pretoken_bytes]
                cnt_pairs[new_prev_pair] = cnt_pairs.get(new_prev_pair, 0)+cnt_pretoken_bytes[pretoken_bytes]
                if new_prev_pair not in pos_pair:
                    pos_pair[new_prev_pair] = []
                pos_pair[new_prev_pair].append((pretoken_idx, prev_idx_start)) # pretoken index, pair index

            nxt_idx_start = pair_idx+len(new_token)
            if nxt_idx_start<len(pretoken_bytes):
                nxt_token = pretoken_bytes[nxt_idx_start:nxt_idx_start+length[nxt_idx_start]]
                nxt_pair = (most_common_pair[1], nxt_token)
                new_nxt_pair = (new_token, nxt_token)
                cnt_pairs[nxt_pair] -= cnt_pretoken_bytes[pretoken_bytes]
                cnt_pairs[new_nxt_pair] = cnt_pairs.get(new_nxt_pair, 0)+cnt_pretoken_bytes[pretoken_bytes]
                if new_nxt_pair not in pos_pair:
                    pos_pair[new_nxt_pair] = []
                pos_pair[new_nxt_pair].append((pretoken_idx, pair_idx))
            
            pair_idx1 = pair_idx+len(most_common_pair[0])
            pair_idx2 = pair_idx+len(new_token)
            length[pair_idx1] = length[pair_idx1-1] = 0
            length[pair_idx] = length[pair_idx2-1] = len(new_token)
        del cnt_pairs[most_common_pair]

    return vocab, merges

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]|None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # appending special_tokens to the vocabulary if they aren’t already there
        self.special_tokens.sort(key=len, reverse=True) # match longest first
        vocab_set = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in vocab_set:
                self.vocab[len(self.vocab)] = token_bytes
        del vocab_set
        
        self.token2idx = {v: k for k, v in vocab.items()}
        self.merges_set = set(merges)

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]|None = None) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges 
        (in the same format that your BPE training code output) and (optionally) a list of special 
        tokens. This method should accept the following additional parameters:

        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (list[str] | None, optional): _description_. Defaults to None.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _encode_pretoken(self, pre_token: str) -> list[int]:
        """Encode a pre-token into a sequence of token IDs using BPE."""
        pre_token = pre_token.encode("utf-8")
        if pre_token in self.token2idx:
            return [self.token2idx[pre_token]]

        # Apply BPE merges to the pre-token
        byte_arr = list(pre_token[i:i+1] for i in range(len(pre_token)))
        for token1, token2 in self.merges:
            if token1 not in byte_arr or token2 not in byte_arr:
                continue
            new_byte_arr = []
            i = 0
            while i < len(byte_arr)-1:
                if byte_arr[i] == token1 and byte_arr[i+1] == token2:
                    new_byte_arr.append(token1+token2)
                    i += 1
                else:
                    new_byte_arr.append(byte_arr[i])
                i += 1
            if i < len(byte_arr):
                new_byte_arr.append(byte_arr[i])
            byte_arr = new_byte_arr
        
        return list(self.token2idx[b] for b in byte_arr)
    
    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a chunk of string into a sequence of token IDs using BPE."""
        ids = []
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            ids.extend(self._encode_pretoken(pre_token))

        return ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        chunks = []
        if len(self.special_tokens)==0:
            chunks = [text]
        else:
            pat_special_token = "|".join(re.escape(s) for s in self.special_tokens)
            last = 0
            for match in re.finditer(pat_special_token, text):
                if match.start()>0:
                    chunks.append(text[last:match.start()])
                chunks.append(match.group())
                last = match.end()
            if last<len(text):
                chunks.append(text[last:])
        
        ids = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes in self.token2idx:
                ids.append(self.token2idx[chunk_bytes])
                continue

            ids.extend(self._encode_chunk(chunk))
        return ids
    
    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily
        yields token IDs. This is required for memory-eﬀicient tokenization of large files that
        we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        return b''.join([self.vocab[token_id] for token_id in tokens]).decode("utf-8", errors="replace")

if __name__ == "__main__":
    # Example usage
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10_000
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    print(f"longest token: {max(vocab.values(), key=len)}")
    with open("tinystories.vocab", "wb") as f:
        pickle.dump(vocab, f)
    with open("tinystories.merges", "wb") as f:
        pickle.dump(merges, f)

    # input_path = "data/owt_train.txt"
    # vocab_size = 32_000
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=vocab_size,
    #     special_tokens=["<|endoftext|>"],
    # )
    # print(f"longest token: {max(vocab.values(), key=len)}")
    # with open("owt.vocab", "wb") as f:
    #     pickle.dump(vocab, f)
    # with open("owt.merges", "wb") as f:
    #     pickle.dump(merges, f)
