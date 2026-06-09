#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


TAG_PATTERN = re.compile(r"<(?P<tag>vidType|textType)>\s*(?P<label>[01])\s*</(?P=tag)>")
LEGACY_PATTERN = re.compile(r"\$(?P<visual>[01])\$(?P<textual>[01])")
LABEL_KEYS = ("visual_safety_label", "textual_safety_label")


def flip_label(value):
    if isinstance(value, bool):
        return int(not value)
    if isinstance(value, int) and value in (0, 1):
        return 1 - value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return str(1 - int(value.strip()))
    return value


def flip_text_labels(text):
    def replace_tag(match):
        return f"<{match.group('tag')}>{1 - int(match.group('label'))}</{match.group('tag')}>"

    text = TAG_PATTERN.sub(replace_tag, text)

    def replace_legacy(match):
        visual = 1 - int(match.group("visual"))
        textual = 1 - int(match.group("textual"))
        return f"${visual}${textual}"

    return LEGACY_PATTERN.sub(replace_legacy, text)


def normalize(value):
    if isinstance(value, dict):
        normalized = {key: normalize(item) for key, item in value.items()}
        for key in LABEL_KEYS:
            if key in normalized:
                normalized[key] = flip_label(normalized[key])
        labels = normalized.get("safety_labels")
        if isinstance(labels, dict):
            for modality in ("visual", "textual"):
                if modality in labels:
                    labels[modality] = flip_label(labels[modality])
        return normalized
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        return flip_text_labels(value)
    return value


def read_records(path):
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must contain a top-level list.")
        return data


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy safety labels (harmful=0, safe=1) to the canonical "
            "VideoSafety-R1 convention (harmful=1, safe=0)."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must differ.")
    write_records(args.output, [normalize(record) for record in read_records(args.input)])


if __name__ == "__main__":
    main()
