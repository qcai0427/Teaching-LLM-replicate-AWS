#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_file_if_present(api: HfApi, repo_id: str, repo_type: str, path: str, target: str):
    file_path = Path(path)
    if not file_path.is_file():
        return
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=target,
        repo_id=repo_id,
        repo_type=repo_type,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--metrics-file", default="")
    parser.add_argument("--conversations-file", default="")
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    artifact_prefix = args.artifact_prefix.strip("/")
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo="",
    )

    upload_file_if_present(
        api,
        args.repo_id,
        args.repo_type,
        args.metrics_file,
        f"{artifact_prefix}/metrics{Path(args.metrics_file).suffix or '.json'}",
    )
    upload_file_if_present(
        api,
        args.repo_id,
        args.repo_type,
        args.conversations_file,
        f"{artifact_prefix}/conversations{Path(args.conversations_file).suffix or '.json'}",
    )
    upload_file_if_present(
        api,
        args.repo_id,
        args.repo_type,
        args.summary_file,
        f"{artifact_prefix}/run_summary{Path(args.summary_file).suffix or '.json'}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
