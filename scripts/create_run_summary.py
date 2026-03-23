#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--train-config-file", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--eval-config-name", default="")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-log", default="")
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--metrics-file", default="")
    parser.add_argument("--conversations-file", default="")
    parser.add_argument("--last-checkpoint", default="")
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--status", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    payload = {
        "job_name": args.job_name,
        "train_config_file": args.train_config_file,
        "train_config_name": args.train_config_name,
        "eval_config_name": args.eval_config_name,
        "save_dir": args.save_dir,
        "model_dir": args.model_dir,
        "server_port": args.server_port,
        "train_log": args.train_log,
        "eval_log": args.eval_log or None,
        "server_log": args.server_log,
        "metrics_file": args.metrics_file or None,
        "conversations_file": args.conversations_file or None,
        "last_checkpoint": args.last_checkpoint or None,
        "hf_repo_id": args.hf_repo_id or None,
        "status": args.status,
        "overrides": args.override,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
