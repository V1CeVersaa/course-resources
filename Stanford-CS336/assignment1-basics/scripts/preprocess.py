import argparse
import json
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
from loguru import logger

from cs336_basics import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode raw text corpora into a numpy memmap-compatible binary file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more text files to tokenize (concatenated in the given order).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output binary file (readable with np.memmap).",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to the tokenizer vocabulary JSON file.",
    )
    parser.add_argument(
        "--merges",
        type=str,
        required=True,
        help="Path to the tokenizer merges TXT file.",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of special tokens, e.g. <|endoftext|>.",
    )
    parser.add_argument(
        "--document-separator",
        type=str,
        default=None,
        help="Optional separator inserted between input files, e.g. <|endoftext|>.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int64"],
        help="Integer dtype used for the output binary file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of tokens buffered before writing; larger saves I/O at the cost of RAM.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000_000,
        help="Log progress every N tokens; set to 0 to disable progress logs.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding used to read input text files.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional metadata JSON output path (defaults to <output>.meta.json).",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable writing metadata JSON.",
    )
    return parser.parse_args()


def _iter_corpus(
    files: Sequence[Path],
    encoding: str,
    separator: str | None,
) -> Iterator[str]:
    num_files = len(files)
    for idx, file_path in enumerate(files):
        logger.info(f"Reading corpus file: {file_path}")
        with file_path.open("r", encoding=encoding) as f:
            yield from f
        if separator is not None and idx != num_files - 1:
            yield separator


def _ensure_integer_dtype(dtype_str: str) -> np.dtype:
    dtype = np.dtype(dtype_str)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"Only integer dtypes are supported, but received {dtype}.")
    return dtype


def _write_tokens(
    tokens: Iterable[int],
    output_path: Path,
    dtype: np.dtype,
    chunk_size: int,
    log_interval: int,
) -> tuple[int, int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    info = np.iinfo(dtype)
    min_value = info.min
    max_value = info.max

    total_tokens = 0
    total_chunks = 0
    next_log_threshold = log_interval if log_interval > 0 else None

    buffer = np.empty(chunk_size, dtype=dtype)

    with output_path.open("wb") as out_f:
        idx = 0
        for token in tokens:
            if token < min_value or token > max_value:
                raise ValueError(
                    f"Token id {token} is outside the valid range [{min_value}, {max_value}] for dtype {dtype}."
                )

            buffer[idx] = token
            idx += 1

            if idx == chunk_size:
                buffer.tofile(out_f)
                total_tokens += idx
                total_chunks += 1
                idx = 0

                if next_log_threshold is not None and total_tokens >= next_log_threshold:
                    logger.info(f"Wrote {total_tokens:,} tokens so far.")
                    next_log_threshold += log_interval

        if idx > 0:
            buffer[:idx].tofile(out_f)
            total_tokens += idx
            total_chunks += 1

            if next_log_threshold is not None:
                while total_tokens >= next_log_threshold:
                    logger.info(f"Wrote {total_tokens:,} tokens so far.")
                    next_log_threshold += log_interval

    return total_tokens, total_chunks


def main() -> None:
    args = parse_args()

    input_files = [Path(path).expanduser() for path in args.input]
    output_path = Path(args.output).expanduser()
    vocab_path = Path(args.vocab).expanduser()
    merges_path = Path(args.merges).expanduser()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not merges_path.is_file():
        raise FileNotFoundError(f"Merges file not found: {merges_path}")

    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=args.special_tokens,
    )
    logger.info("Tokenizer ready.")

    corpus_iter = _iter_corpus(
        files=input_files,
        encoding=args.encoding,
        separator=args.document_separator,
    )
    token_iter = tokenizer.encode_iterable(corpus_iter)

    dtype = _ensure_integer_dtype(args.dtype)
    start_time = time.perf_counter()
    logger.info(f"Writing encoded tokens to {output_path}")
    total_tokens, total_chunks = _write_tokens(
        tokens=token_iter,
        output_path=output_path,
        dtype=dtype,
        chunk_size=args.chunk_size,
        log_interval=args.log_interval,
    )
    elapsed = time.perf_counter() - start_time

    size_bytes = output_path.stat().st_size if output_path.exists() else 0
    logger.info(
        f"Finished writing {total_tokens:,} tokens ({size_bytes / (1024**2):.2f} MiB) in {elapsed:.1f} seconds."
    )

    if total_tokens == 0:
        logger.warning("No tokens were written. Check whether the input files are empty.")

    if not args.no_metadata:
        metadata_path = (
            Path(args.metadata_path).expanduser()
            if args.metadata_path is not None
            else output_path.parent / f"{output_path.name}.meta.json"
        )
        metadata = {
            "num_tokens": total_tokens,
            "dtype": dtype.str,
            "input_files": [str(path) for path in input_files],
            "output_file": str(output_path),
            "tokenizer_vocab": str(vocab_path),
            "tokenizer_merges": str(merges_path),
            "special_tokens": args.special_tokens,
            "document_separator": args.document_separator,
            "chunk_size": args.chunk_size,
            "log_interval": args.log_interval,
            "encoding": args.encoding,
            "size_bytes": size_bytes,
            "num_write_chunks": total_chunks,
            "elapsed_seconds": elapsed,
        }

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
