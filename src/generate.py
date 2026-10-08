"""Generate text from a trained run.

Usage:
    python src/generate.py --run-name baseline --seed "to be or not to be" --words 80 --temperature 0.8
"""
import argparse
import json
from pathlib import Path

import textgen

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--seed", required=True)
    p.add_argument("--words", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    a = p.parse_args()

    from tensorflow import keras
    run_dir = ROOT / "runs" / a.run_name
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    itos, stoi = textgen.load_vocab(run_dir / "vocab.json")
    model = keras.models.load_model(run_dir / "best.keras")
    print(textgen.generate_text(model, a.seed, stoi, itos, cfg["seq_len"],
                                a.words, a.temperature))


if __name__ == "__main__":
    main()
