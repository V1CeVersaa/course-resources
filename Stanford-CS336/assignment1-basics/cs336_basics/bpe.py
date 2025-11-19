import os
import pathlib
import time
from collections import Counter, defaultdict

from loguru import logger
from tqdm import tqdm

from .pretokenization import pretokenize


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # print("")
    # logger.info(f"Parameter input_path = {input_path}")
    # logger.info(f"Parameter vocab_size = {vocab_size}")
    # logger.info(f"Parameter special_tokens = {special_tokens}")
    # logger.info(f"Parameter kwargs = {kwargs}")

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = _get_basic_vocab(special_tokens)
    # logger.info(f"Initialized basic vocabulary with size: {len(vocab)}")

    word_counter: Counter[bytes] = pretokenize(input_path, special_tokens)
    word_symbol: defaultdict[bytes, list[bytes]] = defaultdict(list)
    for word, _ in word_counter.items():
        word_symbol[word] = [bytes([b]) for b in word]

    # logger.info(f"Completed pretokenization. Unique tokens found: {len(word_counter)}")

    pair_occurrences: Counter[tuple[bytes, bytes]] = _get_init_pair_occurrences(word_counter, word_symbol)
    affected_words: dict[tuple[bytes, bytes], set[bytes]] = _build_affected_words(word_symbol)  # inverted index
    # logger.info(f"Initialized pair occurrences and affected words. Unique pairs found: {len(pair_occurrences)}.")

    iter_times = vocab_size - len(vocab)
    for _ in tqdm(range(iter_times), desc="Finding byte pairs to merge"):
        # while len(vocab) < vocab_size:  # stop when the vocabulary size is reached
        if not pair_occurrences:
            logger.warning("No more available pairs. Stop early.")
            break

        merge_pair, _counts = _get_max_pair(pair_occurrences)
        # logger.info(f"Merging pair: {merge_pair} with occurrences: {_counts}")

        merges.append(merge_pair)
        vocab[len(vocab)] = merge_pair[0] + merge_pair[1]

        _update_pair_occurrences(word_counter, word_symbol, pair_occurrences, affected_words, merge_pair)

    # logger.info(f"Completed BPE training. Vocabulary size: {len(vocab)}")

    return vocab, merges


def _get_basic_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {token: bytes([token]) for token in range(256)}

    for token in special_tokens:
        token_bs = token.encode("utf-8")
        if token_bs not in vocab.values():
            vocab[len(vocab)] = token_bs

    return vocab


def _get_init_pair_occurrences(
    word_counter: Counter[bytes], word_symbol: defaultdict[bytes, list[bytes]]
) -> Counter[tuple[bytes, bytes]]:
    pair_occurrences: Counter[tuple[bytes, bytes]] = Counter()

    for word, _ in tqdm(word_counter.items(), desc="Initializing pair occurrences"):
        for i in range(len(word_symbol[word]) - 1):
            pair = (word_symbol[word][i], word_symbol[word][i + 1])
            pair_occurrences[pair] += word_counter[word]

    return pair_occurrences


def _build_affected_words(word_symbol: defaultdict[bytes, list[bytes]]) -> dict[tuple[bytes, bytes], set[bytes]]:
    aw: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)
    for w, syms in tqdm(word_symbol.items(), desc="Building affected words index"):
        for i in range(len(syms) - 1):
            aw[(syms[i], syms[i + 1])].add(w)
    return aw


def _get_max_pair(pair_occurrences: Counter[tuple[bytes, bytes]]) -> tuple[tuple[bytes, bytes], int]:
    best_pair, max_count = max(pair_occurrences.items(), key=lambda kv: (kv[1], kv[0]))
    return best_pair, max_count


def _pairs_counter(syms: list[bytes]) -> Counter[tuple[bytes, bytes]]:
    c: Counter[tuple[bytes, bytes]] = Counter()
    for i in range(len(syms) - 1):
        c[(syms[i], syms[i + 1])] += 1
    return c


def _update_pair_occurrences(
    word_counter: Counter[bytes],
    word_symbol: defaultdict[bytes, list[bytes]],
    pair_occurrences: Counter[tuple[bytes, bytes]],
    affected_words: dict[tuple[bytes, bytes], set[bytes]],
    merge_pair: tuple[bytes, bytes],
):
    if merge_pair not in pair_occurrences:
        raise RuntimeError(f"Pair {merge_pair} to delete does not exist in pair_occurrences")

    a, b = merge_pair
    m = a + b

    words = list(affected_words.get(merge_pair, ()))

    for word in words:
        count = word_counter[word]
        syms = word_symbol[word]

        i = 0
        new_syms: list[bytes] = []
        while i < len(syms):  # build new symbol list with merged pairs
            if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                new_syms.append(m)
                i += 2
            else:
                new_syms.append(syms[i])
                i += 1

        word_symbol[word] = new_syms

        old_pairs = _pairs_counter(syms)
        new_pairs = _pairs_counter(new_syms)

        for p in set(old_pairs.keys()) | set(new_pairs.keys()):  # update pair occurrences
            delta = (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * count  # calculate change
            if delta != 0:
                pair_occurrences[p] += delta
                if pair_occurrences[p] <= 0:
                    pair_occurrences.pop(p, None)

            # update affected words
            if new_pairs.get(p, 0) > 0:
                affected_words.setdefault(p, set()).add(word)
            else:
                s = affected_words.get(p)
                if s is not None:
                    s.discard(word)
                    if not s:
                        affected_words.pop(p, None)

    pair_occurrences.pop(merge_pair, None)
    affected_words.pop(merge_pair, None)


if __name__ == "__main__":
    PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
    input_path = PROJECT_PATH / "data" / "TinyStoriesV2-GPT4-train.txt"
    logger.info(f"input_path: {input_path}")

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    logger.info(f"finished training bpe on dataset TinyStoriesV2, time elapsed {end_time - start_time}")
    logger.info(f"vocabulary info: {len(vocab)}")
    logger.info(f"merge info: {len(merges)}")
