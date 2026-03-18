#!/usr/bin/env python3
import argparse
import re
import sys


ALWAYS_SHOW = [
    re.compile(r"\[start_rl_training\.sh\]"),
    re.compile(r"\[ETA\]"),
    re.compile(r"Training complete!"),
    re.compile(r"TrainOutput\("),
    re.compile(r"Saved conversation table"),
    re.compile(r"Saved metrics"),
    re.compile(r"VLLM server is up"),
    re.compile(r"About to run:"),
    re.compile(r"Done\.$"),
]

ERROR_PATTERNS = [
    re.compile(r"Traceback"),
    re.compile(r"\bERROR\b"),
    re.compile(r"\bError\b"),
    re.compile(r"\bException\b"),
    re.compile(r"CUDA out of memory"),
    re.compile(r"ValueError:"),
    re.compile(r"RuntimeError:"),
    re.compile(r"ModuleNotFoundError:"),
]

TRAIN_PROGRESS = [
    re.compile(r"^\s*\d+%\|"),
    re.compile(r"'train_runtime':"),
    re.compile(r"'loss':"),
]

EVAL_PROGRESS = [
    re.compile(r"Computing metrics"),
    re.compile(r"Metric aggregation:"),
    re.compile(r"Delta mean:"),
    re.compile(r"Initial RM mean:"),
    re.compile(r"End RM mean:"),
    re.compile(r"Leaked solutions mean:"),
    re.compile(r"Does not follow pedagogical values mean:"),
]


def should_show(line: str, mode: str) -> bool:
    patterns = list(ALWAYS_SHOW) + list(ERROR_PATTERNS)
    if mode == "train":
        patterns.extend(TRAIN_PROGRESS)
    else:
        patterns.extend(EVAL_PROGRESS)
        patterns.extend(TRAIN_PROGRESS)
    return any(pattern.search(line) for pattern in patterns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if should_show(line, args.mode):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
