from __future__ import annotations

import random
from pathlib import Path

INPUT = Path("data/train_hard.jsonl")
OUTPUT = Path("data/train_hard_30k.jsonl")
N = 30_000
SEED = 42

lines = [line for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(lines) < N:
    raise SystemExit(f"Not enough records: requested {N}, found {len(lines)}")

rng = random.Random(SEED)
sample = rng.sample(lines, N)
OUTPUT.write_text("\n".join(sample) + "\n", encoding="utf-8")
print(f"wrote {N} records to {OUTPUT}")
