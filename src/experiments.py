"""Bonus: compare architectures / sequence lengths and write runs/results.md.

Usage:  python src/experiments.py
"""
from pathlib import Path

from train import DEFAULT_CFG, ROOT, run

BASE = {**DEFAULT_CFG, "epochs": 15}

EXPERIMENTS = [
    {"run_name": "exp1_baseline"},                          # 1 x LSTM(256), seq 20
    {"run_name": "exp2_seq10", "seq_len": 10},              # shorter context
    {"run_name": "exp3_seq40", "seq_len": 40},              # longer context
    {"run_name": "exp4_deep", "units": [256, 256]},         # 2 stacked LSTMs
    {"run_name": "exp5_wide", "units": [512]},              # wider LSTM
    {"run_name": "exp6_dropout04", "dropout": 0.4},         # more regularisation
]


def main() -> None:
    rows = []
    for override in EXPERIMENTS:
        rows.append(run({**BASE, **override}))

    out = ["| Run | Val loss | Perplexity | Val acc | Epochs | Params | Minutes |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['run_name']} | {r['val_loss']} | {r['perplexity']} | {r['val_acc']} "
                   f"| {r['epochs_run']} | {r['params']:,} | {r['minutes']} |")
    out.append("\n## Sample output (temperature 1.0)\n")
    for r in rows:
        out.append(f"**{r['run_name']}**: {r['sample']}\n")
    path = Path(ROOT / "runs" / "results.md")
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
