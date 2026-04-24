from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_shards(inputs: list[Path], output: Path, meta_output: Path | None, fail_on_duplicate: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    total = 0
    per_file: dict[str, int] = {}

    with output.open("w", encoding="utf-8") as fout:
        for path in inputs:
            count = 0
            with path.open(encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_id = str(record.get("id", f"{path}:{line_no}"))
                    if record_id in seen_ids:
                        duplicate_ids.append(record_id)
                        if fail_on_duplicate:
                            raise ValueError(f"Duplicate id {record_id!r} in {path}")
                    seen_ids.add(record_id)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    total += 1
            per_file[str(path)] = count

    meta = {
        "output": str(output),
        "total_records": total,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids[:100],
        "duplicate_count": len(duplicate_ids),
        "inputs": per_file,
    }
    if meta_output is not None:
        meta_output.parent.mkdir(parents=True, exist_ok=True)
        meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge generated answer JSONL shards into one answer file.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    merge_shards(
        inputs=args.inputs,
        output=args.output,
        meta_output=args.meta_output,
        fail_on_duplicate=not args.allow_duplicates,
    )


if __name__ == "__main__":
    main()
