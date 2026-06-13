"""Smoke-test the EditReward scorer via the running reward service.

Tests the 2-turn history convention:
    history[0] = (prompt, source_image)
    history[1] = (prompt, edited_image)

Usage:
    python3 scripts/test_editreward.py
    python3 scripts/test_editreward.py --url http://host:8080
    python3 scripts/test_editreward.py --source path/to/source.png --edited path/to/edited.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from reward_service.client import RewardClient, RewardRequest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_IMAGE = _REPO_ROOT / "tests" / "assets" / "sample.jpg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--prompt", default="Add a red hat on the person")
    ap.add_argument("--source", type=Path, default=None, help="Source image path")
    ap.add_argument("--edited", type=Path, default=None, help="Edited image path")
    args = ap.parse_args()

    # Use the same image as both source and edited for a basic connectivity test
    source_path = args.source or _DEFAULT_IMAGE
    edited_path = args.edited or _DEFAULT_IMAGE

    if not source_path.exists():
        print(f"ERROR: source image not found: {source_path}", file=sys.stderr)
        return 1
    if not edited_path.exists():
        print(f"ERROR: edited image not found: {edited_path}", file=sys.stderr)
        return 1

    client = RewardClient(args.url)

    # Check health
    health = client.health()
    print(f"Service health: {health}")

    # Build request with 2-turn history
    source_img = Image.open(source_path).convert("RGB")
    edited_img = Image.open(edited_path).convert("RGB")

    req = RewardRequest(
        history=[
            (args.prompt, source_img),
            (args.prompt, edited_img),
        ],
        required_rewards=["editreward"],
    )

    print(f"\nScoring with EditReward:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Source: {source_path}")
    print(f"  Edited: {edited_path}")

    response = client.score([req])

    if response and response[0]:
        scores = response[0]
        print(f"\n  Results:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")
        return 0
    else:
        print("\n  ERROR: No response from service", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
