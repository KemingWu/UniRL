"""Convert pico-banana-400k JSONL to UniRL prompt dataset format.

Input format (per line):
    {"text": "...", "local_input_image": "/path/to/source.jpg",
     "output_image": "/path/to/edited.png", "edit_type": "...", ...}

Output format (per line):
    {"prompt": "...", "media": [{"modality": "image", "role": "condition",
     "uri": "/path/to/source.jpg"}], "metadata": {"edit_type": "..."}}

The output_image is NOT included — during RL training the model generates
its own edited images; the source image is the conditioning input.

Usage:
    python scripts/convert_pico_banana_to_unirl.py \
        --input /path/to/sft_with_local_source_image_path.jsonl \
        --train-output datasets/editreward/train.jsonl \
        --test-output datasets/editreward/test.jsonl \
        --test-size 1000 \
        --max-aspect-ratio 1.3 \
        --seed 42
"""

import argparse
import json
import os
import random

from PIL import Image


def convert_line(raw: dict) -> dict:
    """Convert one raw entry to UniRL prompt example format."""
    prompt = raw.get("text") or raw.get("summarized_text", "")
    source_image = raw["local_input_image"]

    result = {
        "prompt": prompt,
        "media": [{"modality": "image", "role": "condition", "uri": source_image}],
    }

    # Preserve useful metadata
    metadata = {}
    if raw.get("edit_type"):
        metadata["edit_type"] = raw["edit_type"]
    if raw.get("summarized_text"):
        metadata["summarized_text"] = raw["summarized_text"]
    if metadata:
        result["metadata"] = metadata

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--train-output", required=True, help="Output train JSONL")
    parser.add_argument("--test-output", required=True, help="Output test JSONL")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-exists-check",
        action="store_true",
        help="Skip os.path.exists check on source images (fast on network FS)",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=None,
        help="Only keep images with aspect ratio <= this value (e.g. 1.3 for near-square)",
    )
    args = parser.parse_args()

    # Read and convert all lines
    samples = []
    skipped = 0
    skipped_ratio = 0
    with open(args.input, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Skip if source image path is missing
            source_path = raw.get("local_input_image", "")
            if not source_path:
                skipped += 1
                continue

            # Optionally skip slow file existence check
            if not args.skip_exists_check and not os.path.exists(source_path):
                skipped += 1
                continue

            # Aspect ratio filter
            if args.max_aspect_ratio is not None:
                try:
                    img = Image.open(source_path)
                    w, h = img.size
                    ratio = max(w, h) / min(w, h)
                    if ratio > args.max_aspect_ratio:
                        skipped_ratio += 1
                        continue
                except Exception:
                    skipped += 1
                    continue

            converted = convert_line(raw)
            if converted["prompt"]:
                samples.append(converted)
            else:
                skipped += 1

    print(f"Loaded {len(samples)} valid samples, skipped {skipped} (invalid/missing)")
    if args.max_aspect_ratio is not None:
        print(f"Skipped {skipped_ratio} images with aspect ratio > {args.max_aspect_ratio}")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    test_size = min(args.test_size, len(samples) // 10)
    test_samples = samples[:test_size]
    train_samples = samples[test_size:]

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Write outputs
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output), exist_ok=True)

    with open(args.train_output, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(args.test_output, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Written: {args.train_output}, {args.test_output}")


if __name__ == "__main__":
    main()
