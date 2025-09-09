"""Byte-pair Encoding.

Reference:
- Neural Machine Translation of Rare Words with Subword Units
  arXiv:1508.07909

Here we implement a byte-level byte-pair encoding (BBPE), following GPT2.

TODO:
- Prevent BPE from merging across character categories for any byte sequence. [GPT2, p. 4]

"""

import collections


def train(
    vocab_count: int, input: list[str], add_bytes: bool = False, debug: bool = False
):
    """Train a BPE using given words."""

    # Map each token (bytes) to its id.
    tokens: dict[bytes, int] = {}

    if add_bytes:
        # Ensure that we have at least 256 tokens to hold all byte values.
        assert vocab_count >= 256

        # Add all byte values
        for i in range(0, 256):
            tokens[bytes([i])] = i

    # Convert every given word to a list of bytes.
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in input
    ]

    while len(tokens) < vocab_count:
        id = len(tokens)

        stats = collections.Counter()

        # NOTE: `word` is a list of bytes
        for word in words:
            # Iterate over consecutive bytes.
            for pair in zip(word[:-1], word[1:]):
                stats[pair] += 1

        if stats.total() == 0:
            raise Exception(f"There are not enough words! Only {id} tokens are formed.")

        most_common_pair: list[bytes] = stats.most_common(1)[0][0]

        new_token: bytes = most_common_pair[0] + most_common_pair[1]
        tokens[new_token] = id

        if debug:
            print(f"info: most_common_pair: {most_common_pair}")
            print(f"=> new token: {new_token} -- {id}")

        updated_words: list[list[bytes]] = []
        for word in words:
            updated_word: list[bytes] = []
            i: int = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    updated_word.append(new_token)
                    i += 2
                else:
                    updated_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                updated_word.append(word[i])
            updated_words.append(updated_word)
        words = updated_words
    return tokens
