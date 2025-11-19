import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = re.compile(PAT, flags=re.VERSION1)


@dataclass
class _cpc_args:
    path: str | os.PathLike[str]
    start: int
    end: int
    special_tokens: list[str]


def pretokenize(input_path: str | os.PathLike[str], special_tokens: list[str]) -> Counter[bytes]:
    """Finished parallel pre-tokenization using multiple processes. TinyStories using 19 seconds on 16 cores."""
    num_processes = 16
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

    # logger.info(f"Finished running find_chunk_boundaries: len(boundaries) = {len(boundaries)}")
    args = [_cpc_args(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(_count_per_chunk, args))
    # logger.info("Finished counting tokens in all chunks, now aggregating results.")

    counter_str: Counter[str] = Counter()
    for c in results:
        counter_str.update(c)
    # logger.info("Finished aggregating results from all chunks.")

    counter_bytes: Counter[bytes] = Counter({k.encode("utf-8"): v for k, v in counter_str.items()})
    return counter_bytes


def _count_per_chunk(args: _cpc_args) -> Counter[str]:
    special_tokens = args.special_tokens
    counter_str_chunk: Counter[str] = Counter()

    split_pattern = (
        re.compile("|".join(re.escape(s) for s in special_tokens), flags=re.VERSION1) if special_tokens else None
    )

    with open(args.path, "rb") as f:
        f.seek(args.start)
        chunk: str = f.read(args.end - args.start).decode("utf-8", errors="ignore")

    if split_pattern is None:
        counter_str_chunk.update(m.group(0) for m in GPT2_REGEX.finditer(chunk))

    else:
        last = 0
        for sm in split_pattern.finditer(chunk):
            segment = chunk[last : sm.start()]
            counter_str_chunk.update(m.group(0) for m in GPT2_REGEX.finditer(segment))
            last = sm.end()
        tail = chunk[last:]

        counter_str_chunk.update(m.group(0) for m in GPT2_REGEX.finditer(tail))

    return counter_str_chunk


def pretokenize_single(input_path: str | os.PathLike[str], special_tokens: list[str]) -> Counter[bytes]:
    num_processes = 16
    counter_str: Counter[str] = Counter()

    split_pattern = (
        re.compile("|".join(re.escape(s) for s in special_tokens), flags=re.VERSION1) if special_tokens else None
    )

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        # logger.info(f"Finished running find_chunk_boundaries: len(boundaries) = {len(boundaries)}")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            _ = file.seek(start)
            chunk: str = file.read(end - start).decode("utf-8", errors="ignore")

            if split_pattern is None:
                for m in GPT2_REGEX.finditer(chunk):
                    counter_str[m.group(0)] += 1

            else:
                last = 0
                for sm in split_pattern.finditer(chunk):
                    segment = chunk[last : sm.start()]
                    for m in GPT2_REGEX.finditer(segment):
                        counter_str[m.group(0)] += 1
                    last = sm.end()
                tail = chunk[last:]

                for m in GPT2_REGEX.finditer(tail):
                    counter_str[m.group(0)] += 1

    counter_bytes: Counter[bytes] = Counter({k.encode("utf-8"): v for k, v in counter_str.items()})
    return counter_bytes


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    _ = file.seek(0, os.SEEK_END)
    file_size = file.tell()
    _ = file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        _ = file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    ## Usage
    with open("data/test_tokenizer.txt", "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            _ = f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
