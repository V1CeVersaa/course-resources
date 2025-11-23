import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import regex as re
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = re.compile(PAT, flags=re.VERSION1)


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.id_to_token: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []

        if special_tokens is not None:
            max_key = max(vocab.keys())
            for token in special_tokens:
                if token.encode() not in vocab.values():
                    max_key += 1
                    vocab[max_key] = token.encode()

        if special_tokens is not None:
            sorted_special_encoders = sorted(self.special_tokens, key=len, reverse=True)
            self.special_tokens_pattern = re.compile(
                "|".join(re.escape(s) for s in sorted_special_encoders), flags=re.VERSION1
            )
        else:
            self.special_tokens_pattern = None

        self.token_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_file(
        cls, vocab_filepath: str | Path, merges_filepath: str | Path, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath) as f:
            gpt2_vocab = json.load(f)

        gpt2_bpe_merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab, merges, special_tokens)

    @lru_cache(maxsize=256)
    def _bpe_merge(self, token_bytes: bytes) -> list[bytes]:
        syms: list[bytes] = [bytes([b]) for b in token_bytes]

        while True:
            pairs = list(zip(syms[:-1], syms[1:]))
            if not pairs:
                break

            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))

            if self.merge_ranks.get(best_pair, float("inf")) == float("inf"):
                break

            new_syms: list[bytes] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best_pair:
                    new_syms.append(syms[i] + syms[i + 1])
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms

        return syms

    @lru_cache(maxsize=128)
    def encode(self, text: str, progress_bar: bool = False) -> list[int]:
        if self.special_tokens_pattern is None:
            segments: list[tuple[str, str]] = [("text", text)]
        else:
            segments = []
            last = 0
            for match in self.special_tokens_pattern.finditer(text):
                if match.start() > last:
                    segments.append(("text", text[last : match.start()]))
                segments.append(("special", match.group(0)))
                last = match.end()
            if last < len(text):
                segments.append(("text", text[last:]))

        result: list[int] = []
        for kind, value in tqdm(segments, disable=not progress_bar):
            if not value:
                continue

            if kind == "special":
                token_id = self.token_to_id.get(value.encode("utf-8"))
                if token_id is None:
                    raise ValueError(f"Unknown special token: {value!r}")
                result.append(token_id)
                continue

            for token_match in GPT2_REGEX.finditer(value):
                token_bytes = token_match.group(0).encode("utf-8")
                for merged_bytes in self._bpe_merge(token_bytes):
                    token_id = self.token_to_id.get(merged_bytes)
                    if token_id is None:
                        raise KeyError(f"Token bytes {merged_bytes!r} missing from vocab")
                    result.append(token_id)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join([self.id_to_token[id] for id in ids])
        return token_bytes.decode("utf-8", errors="replace")
