def get_tokenizer():
    """
    Create a custom tokenizer based on the `o200k_base` encoding.

    This function clones the base 200k-token vocabulary (`o200k_base`) and
    extends it with additional special tokens for structured communication.

    Parameters used in `tiktoken.Encoding`:
    ---------------------------------------
    - name (str):
        A human-readable identifier for the encoding.
        Example: "o200k_harmony" → lets you distinguish this tokenizer
        from others when debugging or switching encodings.

    - pat_str (str):
        Regex pattern string that defines how raw text is split into
        initial chunks before applying merge rules.
        Example: spaces, punctuation, and characters are split according
        to this regex before being merged into tokens.

    - mergeable_ranks (dict):
        The vocabulary mapping of byte sequences to integer IDs.
        This is the backbone of the tokenizer — it tells which subword
        units exist and their ranks for BPE merges.
        Example: {"t": 100, "he": 101, "the": 102}

    - special_tokens (dict):
        Explicitly reserved tokens with fixed integer IDs.
        These are not learned from text but manually defined.
        Examples:
            "<|startoftext|>": 199998   → Marks the beginning of text
            "<|endoftext|>": 199999     → Marks the end of text
            "<|return|>": 200002        → Could represent a newline
            "<|message|>": 200008       → Custom token for structured output
            "<|call|>": 200012          → Custom token for function calls
            "<|reserved_xxx|>": N       → Reserved slots for future use

        The code also reserves a large block of IDs (200013–201087) so you
        can safely add new tokens later without colliding with existing ones.

    Returns:
    --------
    tokenizer (tiktoken.Encoding):
        A tokenizer object that can encode/decode text using the extended
        vocabulary and special tokens.

    Example Usage:
    --------------
    >>> tokenizer = get_tokenizer()
    >>> tokenizer.encode("Hello world")
    [15339, 1917]   # IDs for "Hello" and "world"

    >>> tokenizer.encode("<|message|> Hello")
    [200008, 15339] # Special token ID for <|message|>, then "Hello"

    >>> tokenizer.decode([200008, 15339])
    "<|message|> Hello"
    """
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer
