from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["FileConfig", "TokenizerConfig"]


class FileConfig(BaseModel):
    file_list: list[str] = Field(default_factory=list)
    out_dir: str
    model_file: str
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
    def model_path(self) -> Path:
        return self.output_dir / self.model_file

    @property
    def merge_path(self) -> Path:
        return self.output_dir / self.merge_file


class TokenizerConfig(BaseModel):
    vocab_size: int
    special_tokens: list[str] = Field(default_factory=list)
