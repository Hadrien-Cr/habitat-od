"""Interactive accept/reject review for the borderline band of
hssd400_cross_vocab_mapping.csv's HSSD400 -> vocab CLIP mappings (see
vocab_constants.THRESHOLD / AUTO_ACCEPT_THRESHOLD - this fills in the blank
"<vocab>_reject" cells left for manual review between those two cutoffs).

Usage:
    common/env_utils/reject.sh <floatmin> <floatmax>

For every (HSSD400 class, vocab) pair whose proximity falls in
[floatmin, floatmax] and whose "<vocab>_reject" cell is still blank, prompts
y (accept -> reject=False), n (reject -> reject=True), or q (quit), writing
the answer back to the CSV immediately after each response - progress is
never lost, safe to interrupt (Ctrl-C) and resume later.
"""
import argparse
import csv
import os

BASE_DIR = os.environ["BASE_DIR"]
CSV_PATH = os.path.join(BASE_DIR, "common", "env_utils", "hssd400_cross_vocab_mapping.csv")
_VOCAB_NAMES = ["HSSD80", "SCANNET200", "NYU40", "MPCAT40", "COCO80"]


def _load_rows(path: str) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _save_rows(path: str, rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("floatmin", type=float)
    ap.add_argument("floatmax", type=float)
    args = ap.parse_args()

    rows = _load_rows(CSV_PATH)
    col_idx = {c: i for i, c in enumerate(rows[0])}
    data_rows = rows[2:]  # skip the header-descriptor row + blank separator row

    to_review = []
    for row_i, row in enumerate(data_rows):
        hssd400_class = row[col_idx["HSSD400"]]
        for vocab in _VOCAB_NAMES:
            if row[col_idx[f"{vocab}_reject"]].strip() != "":
                continue
            try:
                prox = float(row[col_idx[f"{vocab}_proximity"]])
            except ValueError:
                continue
            if args.floatmin <= prox <= args.floatmax:
                to_review.append((row_i, vocab, hssd400_class, row[col_idx[vocab]], prox))

    print(f"{len(to_review)} entries to review in [{args.floatmin}, {args.floatmax}]")

    for row_i, vocab, hssd400_class, target_class, prox in to_review:
        try:
            while True:
                ans = input(
                    f"{hssd400_class!r} (HSSD400) -> {target_class!r} ({vocab})  [prox={prox:.4f}]  accept? (y/n/q): "
                ).strip().lower()
                if ans in ("y", "n", "q"):
                    break
                print("please answer y, n, or q (quit)")
        except (EOFError, KeyboardInterrupt):
            print("\nstopping - progress saved.")
            break

        if ans == "q":
            print("stopping - progress saved.")
            break

        data_rows[row_i][col_idx[f"{vocab}_reject"]] = "False" if ans == "y" else "True"
        rows[2:] = data_rows
        _save_rows(CSV_PATH, rows)

    print("done.")


if __name__ == "__main__":
    main()
