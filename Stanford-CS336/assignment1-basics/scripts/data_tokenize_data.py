from pathlib import Path

import numpy as np
import yaml
from config_setting import TokenizerBuildConfig
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from cs336_basics import Tokenizer


class Config(BaseModel):
    tokenizer: TokenizerBuildConfig
    files: list[str] = Field(default_factory=list)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


if __name__ == "__main__":
    logger.info(f"load config from config/conf_{Path(__file__).stem}.yaml")
    cfg = load_config(f"config/conf_{Path(__file__).stem}.yaml")

    vocab: Path = cfg.tokenizer.vocab_path
    merge: Path = cfg.tokenizer.merge_path
    specs: list[str] = cfg.tokenizer.special_tokens

    tokenizer = Tokenizer.from_file(vocab, merge, specs)

    files: list[str] = cfg.files
    for file in files:
        file = Path(file)
        logger.info(f"Tokenizing File {file}")
        with open(file) as f:
            text = f.read()

        encoded = tokenizer.encode(text, progress_bar=True)

        total_batches = 1024
        batch_size = len(encoded) // total_batches
        arr = np.memmap(file.with_suffix(".bin"), dtype=np.uint16, mode="w+", shape=(len(encoded),))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {file.stem}.bin"):
            batch = encoded[idx : idx + batch_size]
            arr[idx : idx + batch_size] = batch
            idx += batch_size
        arr.flush()
