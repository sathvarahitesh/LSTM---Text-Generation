"""Shared helpers: preprocessing, vocabulary, sequence building, model, generation."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


# ----------------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase and strip punctuation (apostrophes inside words are kept: "o'er")."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)             # drop punctuation
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)   # drop stray quote marks
    return re.sub(r"\s+", " ", text).strip()


def load_words(path: str, max_words: int | None = None) -> list[str]:
    """Read the file, clean it and tokenize into a list of words."""
    words = clean_text(Path(path).read_text(encoding="utf-8")).split()
    return words[:max_words] if max_words else words


def build_vocab(words: list[str], max_vocab: int) -> tuple[list[str], dict[str, int]]:
    """Keep the (max_vocab - 2) most frequent words; the rest map to <unk>."""
    common = [w for w, _ in Counter(words).most_common(max_vocab - 2)]
    itos = [PAD, UNK] + common
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode(words: list[str], stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi.get(w, UNK_ID) for w in words], dtype=np.int32)


def make_sequences(ids: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: X = seq_len tokens, y = the token that follows them."""
    windows = np.lib.stride_tricks.sliding_window_view(ids, seq_len + 1)
    return windows[:, :-1].copy(), windows[:, -1].copy()


def save_vocab(itos: list[str], path: Path) -> None:
    path.write_text(json.dumps(itos), encoding="utf-8")


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    itos = json.loads(path.read_text(encoding="utf-8"))
    return itos, {w: i for i, w in enumerate(itos)}


# ----------------------------------------------------------------------------
# 2. Model
# ----------------------------------------------------------------------------
def build_model(vocab_size: int, seq_len: int, embed_dim: int,
                units: list[int], dropout: float, lr: float):
    """Embedding -> stacked LSTM(+Dropout) -> Dense softmax over the vocabulary."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_len,), dtype="int32"),
        layers.Embedding(vocab_size, embed_dim),
    ])
    for i, n in enumerate(units):
        # every LSTM except the last must return the full sequence for stacking
        model.add(layers.LSTM(n, return_sequences=i < len(units) - 1))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(vocab_size, activation="softmax"))
    model.compile(
        # sparse_categorical_crossentropy == categorical crossentropy with integer
        # labels; avoids a (samples x vocab) one-hot matrix that would not fit in RAM.
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------------
# 3. Generation
# ----------------------------------------------------------------------------
def sample_next(probs: np.ndarray, temperature: float) -> int:
    """Temperature sampling: <1 = safer/repetitive, >1 = more random."""
    probs = probs.astype("float64").copy()
    probs[[PAD_ID, UNK_ID]] = 0.0                      # never emit <pad>/<unk>
    logits = np.log(probs + 1e-9) / max(temperature, 1e-3)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


def generate_text(model, seed: str, stoi: dict[str, int], itos: list[str],
                  seq_len: int, n_words: int = 50, temperature: float = 0.8) -> str:
    """Feed the seed, predict one word, append it, repeat n_words times."""
    words = clean_text(seed).split()
    ids = [stoi.get(w, UNK_ID) for w in words]
    for _ in range(n_words):
        window = ids[-seq_len:]
        window = [PAD_ID] * (seq_len - len(window)) + window    # left-pad short seeds
        probs = model(np.array([window], dtype="int32"), training=False).numpy()[0]
        ids.append(sample_next(probs, temperature))
    return " ".join(words + [itos[i] for i in ids[len(words):]])
