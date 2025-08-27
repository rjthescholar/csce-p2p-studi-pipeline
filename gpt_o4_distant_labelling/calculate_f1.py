#!/usr/bin/env python
"""
evaluate_concepts.py
--------------------
Compute precision, recall, F1, and error buckets for a GPT-generated
concept list against a human (gold) list.

Matching rule
-------------
* Exact string equality after LOWER-CASING each line.
* One concept per line in each file.

CLI usage
---------
$ python evaluate_concepts.py gpt_output.txt human_gold.txt

The script prints a metric summary and the three buckets:

    === METRICS ===
    true_positives   : 18
    false_positives  : 4
    false_negatives  : 2
    precision        : 0.8182
    recall           : 0.9
    f1               : 0.8571
    accuracy         : 0.9

    TRUE POSITIVES (18):
       • ai ethics
       • ...

    FALSE POSITIVES (4):
       • digital twin
       • ...

    FALSE NEGATIVES (2):
       • privacy by design
       • ...
"""

import argparse
from pathlib import Path
from typing import Dict, Set, Union


# --------------------------------------------------------------------------- #
# Core evaluation logic
# --------------------------------------------------------------------------- #
def _load_concepts(path: Union[str, Path]) -> Set[str]:
    """Load a text file → lower-cased, deduplicated set of concepts."""
    with Path(path).open(encoding="utf-8") as f:
        return {line.rstrip("\n").lower() for line in f if line.strip()}


def evaluate_concept_files(
    pred_file: Union[str, Path], gold_file: Union[str, Path]
) -> Dict[str, float]:
    """Return metrics dict and print buckets/summary to stdout."""
    pred = _load_concepts(pred_file)
    gold = _load_concepts(gold_file)

    true_pos = pred & gold
    false_pos = pred - gold
    false_neg = gold - pred

    tp, fp, fn = len(true_pos), len(false_pos), len(false_neg)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = tp / len(gold) if gold else 0.0

    metrics: Dict[str, float] = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }

    # -------- pretty print --------------------------------------------------
    print("=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k:17}: {v}")

    def _print_bucket(name: str, bucket: Set[str]) -> None:
        print(f"\n{name} ({len(bucket)}):")
        for concept in sorted(bucket):
            print("   •", concept)

    _print_bucket("TRUE POSITIVES", true_pos)
    _print_bucket("FALSE POSITIVES", false_pos)
    _print_bucket("FALSE NEGATIVES", false_neg)

    return metrics


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate GPT concept extraction against human gold."
    )
    p.add_argument(
        "pred_file",
        type=Path,
        help="File with GPT-generated concepts (one per line).",
    )
    p.add_argument(
        "gold_file",
        type=Path,
        help="File with human gold concepts (one per line).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_concept_files(args.pred_file, args.gold_file)
