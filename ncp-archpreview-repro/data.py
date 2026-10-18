import numpy as np

# Synthetic chunk-structured token stream. 6 fixed length-4 "chunk patterns"
# built from 12 distinct token ids, chained by a noisy Markov process over
# chunk identity (next chunk = current+1 with prob 0.8, else uniform random).
#
# Deliberately designed so that WITHIN a chunk the next token is fully
# determined by the current token (a window=1 bigram model solves it), but
# AT a chunk boundary the next token depends on which chunk comes next --
# information a window=1 token encoder cannot see, but that a concept-level
# pathway summarizing the previous chunk can learn to predict.

VOCAB_SIZE = 12
CHUNK_LEN = 4
CHUNKS = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [3, 2, 1, 0],
    [7, 6, 5, 4],
    [11, 10, 9, 8],
]
NUM_CHUNK_TYPES = len(CHUNKS)


def generate_sequence(n_chunks=400, seed=0):
    rng = np.random.default_rng(seed)
    chunk_ids = [rng.integers(0, NUM_CHUNK_TYPES)]
    for _ in range(n_chunks - 1):
        if rng.random() < 0.8:
            nxt = (chunk_ids[-1] + 1) % NUM_CHUNK_TYPES
        else:
            nxt = rng.integers(0, NUM_CHUNK_TYPES)
        chunk_ids.append(nxt)
    tokens = []
    for cid in chunk_ids:
        tokens.extend(CHUNKS[cid])
    return np.array(tokens, dtype=np.int64)


def boundary_positions(n_tokens):
    """Positions t (0-indexed) whose NEXT token (t+1) is the first token of
    a new chunk -- i.e. where prediction genuinely requires chunk-transition
    knowledge, not just the current token."""
    return np.arange(CHUNK_LEN - 1, n_tokens - 1, CHUNK_LEN)
