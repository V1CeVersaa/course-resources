import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = re.compile(PAT, flags=re.VERSION1)


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
        with open(vocab_filepath) as f:
            vocab_data: dict[Any, str | bytes] = json.load(f)

        vocab = {int(k): v.encode("utf-8") if isinstance(v, str) else v for k, v in vocab_data.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0].encode(), parts[1].encode()))

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
    def encode(self, text: str) -> list[int]:
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
        for kind, value in segments:
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
