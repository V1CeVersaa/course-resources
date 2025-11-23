from pathlib import Path

import torch
from pydantic import BaseModel, Field

__all__ = [
    "FileConfig",
    "TokenizerConfig",
    "TokenizerBuildConfig",
    "DatasetConfig",
    "ModelConfig",
    "OptimizerConfig",
    "CheckLogConfig",
    "GenerationConfig",
]


class FileConfig(BaseModel):
    file_list: list[str] = Field(default_factory=list)
    out_dir: str
    vocab_file: str
    merge_file: str

    @property
    def output_dir(self) -> Path:
        path = Path(self.out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vocab_path(self) -> Path:
        return self.output_dir / self.vocab_file

    @property
    def merge_path(self) -> Path:
        return self.output_dir / self.merge_file


class TokenizerConfig(BaseModel):
    vocab_size: int
    special_tokens: list[str] = Field(default_factory=list)


class TokenizerBuildConfig(BaseModel):
    in_dir: str
    vocab_file: str
    merge_file: str
    special_tokens: list[str] = Field(default_factory=list)

    @property
    def input_dir(self) -> Path:
        path = Path(self.in_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vocab_path(self) -> Path:
        return self.input_dir / self.vocab_file

    @property
    def merge_path(self) -> Path:
        return self.input_dir / self.merge_file


class DatasetConfig(BaseModel):
    dataset: str
    context_length: int = Field(default=256)
    batch_size: int = Field(default=64)
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")


class ModelConfig(BaseModel):
    vocab_size: int = Field(default=10000)
    context_length: int = Field(default=256)
    num_layers: int = Field(default=12)
    d_model: int = Field(default=768)
    num_heads: int = Field(default=12)
    d_ff: int = Field(default=3072)
    rope_theta: float = Field(default=10000.0)


class OptimizerConfig(BaseModel):
    total_iters: int
    warmup_iters: int
    lr_max: float
    lr_min: float
    weight_decay: float


class CheckLogConfig(BaseModel):
    check_dir: str
    save_interval: int
    log_interval: int
    eval_interval: int
    eval_iters: int
    wandb_logging: bool = False
    wandb_project: str = "cs336-assignment1"
    wandb_run_name: str = "transformer-lm"

    @property
    def checkpoint_dir(self) -> Path:
        path = Path(self.check_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


class GenerationConfig(BaseModel):
    checkpoint_path: str
    tokenizer_vocab: str
    tokenizer_merges: str
    tokenizer_specs: list[str] = Field(default_factory=list)
    prompt: str = ""
    max_length: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
