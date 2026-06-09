#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


SAFE_LABELS = {"visual": 0, "textual": 0}


def read_records(path):
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a top-level list.")
        return records


def sample(records, count, rng, name):
    if len(records) < count:
        raise ValueError(f"{name} contains {len(records)} records; {count} required.")
    return rng.sample(records, count)


def add_safe_labels(records):
    for record in records:
        record.setdefault("safety_labels", SAFE_LABELS.copy())
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Build the 10k AT-SFT mixture: VST-6k + VCG-plus-2k + LLaVA-2k."
    )
    parser.add_argument("--vst", type=Path, required=True)
    parser.add_argument("--vcg", type=Path, required=True)
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vst-count", type=int, default=6000)
    parser.add_argument("--vcg-count", type=int, default=2000)
    parser.add_argument("--llava-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = sample(read_records(args.vst), args.vst_count, rng, "VST-SFT")
    records += add_safe_labels(
        sample(read_records(args.vcg), args.vcg_count, rng, "VCG-plus")
    )
    records += add_safe_labels(
        sample(read_records(args.llava), args.llava_count, rng, "LLaVA-SFT")
    )
    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
