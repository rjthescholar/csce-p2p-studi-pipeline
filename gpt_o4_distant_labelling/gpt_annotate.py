#!/usr/bin/env python
"""
Annotate every *.txt slide deck in a course folder with OpenAI o3, using a
rolling “prior-knowledge” window:

    • No prior knowledge for the first <window> lectures.
    • Lecture N sees only concepts that are ≥ window + 1 lectures older.
      (window = 4  →  L6 gets L1 concepts, L7 gets L1-2, etc.)

Results are written next to each deck as  <Lecture>.concepts.txt.
Existing .concepts.txt files are reused unless --overwrite is supplied.

Typical usage
-------------
$ export OPENAI_API_KEY="sk-…"      # or pass --api-key
$ python annotate_course.py
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Set

from openai import OpenAI


# ───────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────
def natural_key(path: Path) -> List[int | str]:
    """Sort “Lecture 2” before “Lecture 10” (numbers treated as ints)."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def dedupe_preserve_case(existing: List[str], new_terms: List[str]) -> None:
    """
    Extend *existing* with *new_terms*, ignoring case-insensitive duplicates.
    Modifies the list in place.
    """
    seen: Set[str] = {t.lower() for t in existing}
    for term in new_terms:
        if term.lower() not in seen:
            existing.append(term)
            seen.add(term.lower())


# ───────────────────────────────────────────────────────────
# per-deck call
# ───────────────────────────────────────────────────────────
USER_TMPL = """Here is the deck text:

[[START]]
{deck_text}
[[END]]"""

PRIOR_MSG_TMPL = (
    "The following newline-separated terms are prior knowledge. "
    "When you compile the concept list, EXCLUDE any term that matches "
    "(case-insensitive):\n{terms}"
)


def annotate_deck(
    deck_path: Path,
    client: OpenAI,
    model: str,
    primer_text: str,
    prior_terms: List[str],
) -> List[str]:
    """Return a list of concepts for one slide deck."""
    deck_text = deck_path.read_text(encoding="utf-8", errors="ignore")

    # build chat
    messages = [{"role": "system", "content": primer_text}]
    if prior_terms:
        messages.append(
            {
                "role": "system",
                "content": PRIOR_MSG_TMPL.format(terms="\n".join(prior_terms)),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": USER_TMPL.format(deck_text=deck_text),
        }
    )

    # call model
    resp = client.chat.completions.create(model=model, messages=messages)
    return [c.strip() for c in resp.choices[0].message.content.splitlines() if c.strip()]


# ───────────────────────────────────────────────────────────
# main loop
# ───────────────────────────────────────────────────────────
def annotate_course(
    course_dir: Path,
    output_dir: Path,
    primer_txt: Path,
    api_key: str,
    model: str = "o3",
    window: int = 4,
    overwrite: bool = False,
) -> None:
    """Iterate through every .txt deck in *course_dir* and annotate it."""
    txt_files = sorted(course_dir.glob("*.txt"), key=natural_key)
    if not txt_files:
        print(f"[ERROR] No .txt files found in {course_dir.resolve()}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    primer_text = primer_txt.read_text(encoding="utf-8", errors="ignore")
    client = OpenAI(api_key=api_key)

    concepts_by_lecture: List[List[str]] = []  # keeps timeline

    def prior_terms_for(idx: int) -> List[str]:
        """Concepts from lectures older than *window*."""
        cutoff = idx - window
        if cutoff < 0:
            return []
        prior: List[str] = []
        for older in concepts_by_lecture[:cutoff]:
            dedupe_preserve_case(prior, older)
        return prior

    for idx, txt in enumerate(txt_files):
        out_path = output_dir / f"{txt.stem}.concepts.txt"

        if out_path.exists() and not overwrite:
            existing = [
                l.rstrip("\n")
                for l in out_path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            concepts_by_lecture.append(existing)
            print(f"[SKIP] {out_path.name} exists, reused for prior knowledge.")
            continue

        concepts = annotate_deck(
            deck_path=txt,
            client=client,
            model=model,
            primer_text=primer_text,
            prior_terms=prior_terms_for(idx),
        )
        out_path.write_text("\n".join(concepts), encoding="utf-8")
        print(f"[OK]   {txt.name}: {len(concepts)} concepts → {out_path.name}")

        concepts_by_lecture.append(concepts)


# ───────────────────────────────────────────────────────────
# CLI entry point
# ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate a course folder of *.txt slide decks with OpenAI o3."
    )
    p.add_argument(
        "--course-dir",
        type=Path,
        default=Path("courses_to_annotate/data_text/CS-1550 Lecture Slides"),
        help="Folder containing .txt slide decks.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("courses_to_annotate/data_text/CS-1550 Lecture Slides_output"),
        help="Folder to write *.concepts.txt files.",
    )
    p.add_argument(
        "--primer-txt",
        type=Path,
        default=Path("primer.txt"),
        help="System prompt file.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=4,
        help="Number of most recent lectures to exclude from prior knowledge.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-annotate lectures even if *.concepts.txt already exists.",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI key (defaults to env var OPENAI_API_KEY).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        sys.exit("Provide an OpenAI API key via --api-key or OPENAI_API_KEY env var.")

    annotate_course(
        course_dir=args.course_dir,
        output_dir=args.output_dir,
        primer_txt=args.primer_txt,
        api_key=args.api_key,
        window=args.window,
        overwrite=args.overwrite,
    )
