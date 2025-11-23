import json
from pathlib import Path

import yaml
from config_setting import FileConfig, TokenizerConfig
from loguru import logger
from pydantic import BaseModel

from cs336_basics import train_bpe
from cs336_basics.tokenizer import gpt2_bytes_to_unicode


class Config(BaseModel):
    file: FileConfig
    tokenizer: TokenizerConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def main():
    logger.info(f"load config from config/conf_{Path(__file__).stem}.yaml")
    cfg = load_config(f"config/conf_{Path(__file__).stem}.yaml")

    # currently only support input one training file, which is not like huggingface tokenizer
    training_file = cfg.file.file_list[0]
    logger.info(f"load training file {training_file}")

    vocab_size = cfg.tokenizer.vocab_size
    spec_token = cfg.tokenizer.special_tokens

    vocab, merges = train_bpe(training_file, vocab_size, spec_token)
    logger.info(f"finished training bpe tokenizer with vocab_size={vocab_size} and special_tokens={spec_token}")

    vocab_file = cfg.file.vocab_path
    merge_file = cfg.file.merge_path

    byte_to_unicode = gpt2_bytes_to_unicode()

    # Convert the byte tokens in the vocab back to string tokens using the unicode mapping
    # vocab is Dict[int, bytes]
    reversed_vocab = {"".join([byte_to_unicode[b] for b in bytes_token]): k for k, bytes_token in vocab.items()}

    # Convert the byte sequences in merges back to string tokens
    reversed_merges = [
        " ".join([
            "".join([byte_to_unicode[b] for b in merge[0]]),
            "".join([byte_to_unicode[b] for b in merge[1]]),
        ])
        for merge in merges
    ]

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(reversed_vocab, f, ensure_ascii=False, indent=4)

    with open(merge_file, "w", encoding="utf-8") as f:
        for merge in reversed_merges:
            f.write(merge + "\n")

    logger.info("vocab and merges data saved")


if __name__ == "__main__":
    main()
