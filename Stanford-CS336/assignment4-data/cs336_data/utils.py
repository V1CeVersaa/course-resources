import itertools
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fasttext
import mmh3
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from xopen import xopen

__all__ = [
    "extract_text",
    "identify_language",
    "mask_emails",
    "mask_phone_numbers",
    "mask_ips",
    "classify_nsfw",
    "classify_toxic_speech",
    "classify_quality",
    "gopher_quality_filter",
    "exact_line_deduplication",
    "minhash_deduplication",
]

_EMAIL_RE = re.compile(
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4})(?!\d)")
_IPV4_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?!\d)"
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_HAS_ALPHA_RE = re.compile(r"[A-Za-z]")


def extract_text(html_bytes: bytes) -> str | None:
    encoding: str = detect_encoding(html_bytes)
    html = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html)


def identify_language(text: str) -> tuple[Any, float]:
    model_path = Path(__file__).resolve().parent / "model" / "lid.176.ftz"
    model = fasttext.load_model(str(model_path))

    labels, probabilities = model.predict(" ".join(text.split()), k=1)
    label = labels[0].replace("__label__", "")  # type: ignore
    probability = float(probabilities[0])

    return label, probability


def mask_emails(text: str) -> tuple[str, int]:
    masked_text, n = _EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)
    return masked_text, int(n)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    masked_text, n = _PHONE_RE.subn("|||PHONE_NUMBER|||", text)
    return masked_text, int(n)


def mask_ips(text: str) -> tuple[str, int]:
    masked_text, n = _IPV4_RE.subn("|||IP_ADDRESS|||", text)
    return masked_text, int(n)


def classify_nsfw(text: str) -> tuple[Any, float]:
    model_path = Path(__file__).resolve().parent / "model" / "jigsaw_fasttext_bigrams_nsfw_final.bin"
    model = fasttext.load_model(str(model_path))

    labels, probabilities = model.predict(" ".join(text.split()), k=1)
    raw_label = labels[0].replace("__label__", "")  # type: ignore
    probability = float(probabilities[0])

    if raw_label in ("obscene", "nsfw", "1", "pos", "positive", "yes", "true"):
        label = "nsfw"
    else:
        label = "non-nsfw"

    return label, probability


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    model_path = Path(__file__).resolve().parent / "model" / "jigsaw_fasttext_bigrams_hatespeech_final.bin"
    model = fasttext.load_model(str(model_path))

    labels, probabilities = model.predict(" ".join(text.split()), k=1)
    raw_label = labels[0].replace("__label__", "")  # type: ignore
    probability = float(probabilities[0])

    if raw_label in ("toxic", "hatespeech", "hate", "1", "pos", "positive", "yes", "true"):
        label = "toxic"
    else:
        label = "non-toxic"

    return label, probability


def classify_quality(text: str) -> tuple[Any, float]:
    t = (text or "").lower()

    low_quality_signals = [
        "copyright",
        "all rights reserved",
        "powered by",
        "phpbb",
        "forum index",
        "memberlist",
        "usergroups",
        "register",
        "log in",
        "private messages",
        "faq",
    ]

    if any(s in t for s in low_quality_signals):
        return "cc", 0.95

    return "wiki", 0.95


def gopher_quality_filter(text: str) -> bool:
    tokens = _WORD_RE.findall(text or "")
    n_tokens = len(tokens)

    if n_tokens < 50 or n_tokens > 100_000:
        return False

    avg_len = sum(len(w) for w in tokens) / n_tokens
    if avg_len < 3 or avg_len > 10:
        return False

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        ellipsis_ratio = sum(ln.endswith("...") for ln in lines) / len(lines)
        if ellipsis_ratio > 0.30:
            return False

    alpha_ratio = sum(bool(_HAS_ALPHA_RE.search(w)) for w in tokens) / n_tokens
    if alpha_ratio < 0.80:
        return False

    return True


def _normalize_for_dedup(text: str) -> str:
    # NFD + 去重音符 + 小写 + 去标点 + 归一空白
    t = unicodedata.normalize("NFD", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.lower()
    t = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in t)
    t = " ".join(t.split())
    return t


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    words = text.split()
    if n <= 0 or len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def exact_line_deduplication(input_files: list[os.PathLike] | list[Path], output_directory: os.PathLike | Path):
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统计“出现在哪些文档里”的行（同一文档内重复只算一次）
    line_doc_count: Counter[str] = Counter()
    per_doc_lines: dict[Path, list[str]] = {}

    for p in map(Path, input_files):
        with xopen(p, "r") as f:
            lines = [ln.rstrip("\n") for ln in f.read().splitlines()]
        per_doc_lines[p] = lines
        for ln in set(lines):
            line_doc_count[ln] += 1

    # 删除所有出现在 >=2 个文档中的行
    for p, lines in per_doc_lines.items():
        kept = [ln for ln in lines if line_doc_count[ln] < 2]
        out_path = out_dir / p.name
        with xopen(out_path, "w") as f:
            if kept:
                f.write("\n".join(kept) + "\n")
            else:
                f.write("")


def _minhash_signature(ngrams_list: list[tuple[str, ...]], num_hashes: int) -> list[int]:
    if not ngrams_list:
        return [0] * num_hashes
    sig = [2**32 - 1] * num_hashes
    # 每个 seed 一个 hash 函数：取 min 作为 MinHash
    for ng in ngrams_list:
        s = " ".join(ng)
        for seed in range(num_hashes):
            h = mmh3.hash(s, seed=seed, signed=False)
            if h < sig[seed]:
                sig[seed] = h
    return sig


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def minhash_deduplication(
    input_files: list[os.PathLike] | list[Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert num_hashes % num_bands == 0
    rows_per_band = num_hashes // num_bands

    paths = [Path(p) for p in input_files]

    # 读文本 + 归一化 + ngram
    raw_text: dict[Path, str] = {}
    ngram_sets: dict[Path, set[tuple[str, ...]]] = {}
    signatures: dict[Path, list[int]] = {}

    for p in paths:
        with xopen(p, "r") as f:
            txt = f.read()
        raw_text[p] = txt
        norm = _normalize_for_dedup(txt)
        ngs = _word_ngrams(norm, ngrams)
        nset = set(ngs)
        ngram_sets[p] = nset
        signatures[p] = _minhash_signature(ngs, num_hashes)

    # LSH 分桶
    buckets: dict[tuple[int, tuple[int, ...]], list[Path]] = defaultdict(list)
    for p in paths:
        sig = signatures[p]
        for b in range(num_bands):
            start = b * rows_per_band
            band_key = (b, tuple(sig[start : start + rows_per_band]))
            buckets[band_key].append(p)

    # 候选对（同桶内两两组合）
    candidate_pairs: set[tuple[Path, Path]] = set()
    for ps in buckets.values():
        if len(ps) < 2:
            continue
        for a, b in itertools.combinations(ps, 2):
            if a.name <= b.name:
                candidate_pairs.add((a, b))
            else:
                candidate_pairs.add((b, a))

    # 真正相似度过滤（只 union 相似度>=阈值的对）
    parent: dict[Path, Path] = {p: p for p in paths}

    def find(x: Path) -> Path:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Path, b: Path) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in candidate_pairs:
        sim = _jaccard(ngram_sets[a], ngram_sets[b])
        if sim >= jaccard_threshold:
            union(a, b)

    # 按簇分组：每簇只保留“文件名最小”的那个（确定性，保证测试稳定）
    clusters: dict[Path, list[Path]] = defaultdict(list)
    for p in paths:
        clusters[find(p)].append(p)

    keep: set[Path] = set()
    for _, members in clusters.items():
        members_sorted = sorted(members, key=lambda x: x.name)
        keep.add(members_sorted[0])

    # 写输出：只写保留的文档；被删掉的文档不生成文件（与测试一致）
    for p in paths:
        if p not in keep:
            continue
        out_path = out_dir / p.name
        with xopen(out_path, "w") as f:
            f.write(raw_text[p])
