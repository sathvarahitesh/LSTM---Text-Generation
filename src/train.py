"""Train an LSTM word-level language model and write samples.

Usage (from project root):
    python src/train.py --run-name baseline
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import textgen

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CFG = {
    "run_name": "baseline",
    "data": str(ROOT / "data" / "shakespeare.txt"),
    "max_words": 300_000,     # cap corpus size so CPU training stays fast (0 = all)
    "max_vocab": 8000,
    "seq_len": 20,
    "embed_dim": 100,
    "units": [256],
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 256,
    "epochs": 30,
    "val_split": 0.1,
    "patience": 3,
    "seed": 42,
}

SAMPLE_SEEDS = ["to be or not to be", "o romeo romeo", "the king", "my lord"]


def run(cfg: dict) -> dict:
    """Full pipeline: preprocess -> train -> save -> generate samples."""
    import tensorflow as tf
    from tensorflow import keras

    np.random.seed(cfg["seed"])
    tf.random.set_seed(cfg["seed"])

    run_dir = ROOT / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    words = textgen.load_words(cfg["data"], cfg["max_words"] or None)
    itos, stoi = textgen.build_vocab(words, cfg["max_vocab"])
    ids = textgen.encode(words, stoi)
    X, y = textgen.make_sequences(ids, cfg["seq_len"])

    # Chronological split: random splitting would leak overlapping windows into validation.
    split = int(len(X) * (1 - cfg["val_split"]))
    X_tr, y_tr, X_val, y_val = X[:split], y[:split], X[split:], y[split:]
    print(f"[{cfg['run_name']}] words={len(words):,} vocab={len(itos):,} "
          f"train={len(X_tr):,} val={len(X_val):,}")

    # 2. Model
    model = textgen.build_model(len(itos), cfg["seq_len"], cfg["embed_dim"],
                                cfg["units"], cfg["dropout"], cfg["lr"])
    model.summary()

    # 3. Train with checkpointing + early stopping
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]
    start = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     batch_size=cfg["batch_size"], epochs=cfg["epochs"],
                     callbacks=callbacks, verbose=2)
    minutes = (time.time() - start) / 60

    best = int(np.argmin(hist.history["val_loss"]))
    val_loss = float(hist.history["val_loss"][best])
    metrics = {
        "run_name": cfg["run_name"],
        "val_loss": round(val_loss, 4),
        "perplexity": round(float(np.exp(val_loss)), 1),
        "val_acc": round(float(hist.history["val_accuracy"][best]), 4),
        "epochs_run": len(hist.history["loss"]),
        "params": int(model.count_params()),
        "minutes": round(minutes, 1),
    }

    # 4. Save artifacts
    textgen.save_vocab(itos, run_dir / "vocab.json")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in hist.history.items()}),
        encoding="utf-8")

    # 5. Generate sample outputs
    seeds = [" ".join(words[:5])] + SAMPLE_SEEDS     # first words of the dataset + custom
    lines = []
    for seed in seeds:
        for temp in (0.5, 1.0):
            text = textgen.generate_text(model, seed, stoi, itos, cfg["seq_len"], 50, temp)
            lines.append(f"SEED: {seed!r} | temperature={temp}\n{text}\n")
    sample_text = "\n".join(lines)
    (run_dir / "samples.txt").write_text(sample_text, encoding="utf-8")
    metrics["sample"] = lines[1].split("\n", 1)[1].strip()   # one temp=1.0 sample
    print(sample_text)
    print(metrics)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM text generator")
    for key, val in DEFAULT_CFG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, list):
            p.add_argument(flag, type=int, nargs="+", default=val)
        else:
            p.add_argument(flag, type=type(val), default=val)
    run(vars(p.parse_args()))


if __name__ == "__main__":
    main()
