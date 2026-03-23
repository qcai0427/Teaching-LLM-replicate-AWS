#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from config.eval import EvalConfig
from config.train_rl_model import RLModelTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser


def load_train_cfg(config_name: str, overrides: list[str]):
    default_config = OmegaConf.structured(RLModelTrainingConfig)
    loaded = OmegaConf.load(Path("config/train_rl") / f"{config_name}.yaml")
    return OmegaConf.merge(default_config, loaded, OmegaConf.from_dotlist(overrides))


def load_eval_cfg(config_name: str, overrides: list[str]):
    default_config = OmegaConf.structured(EvalConfig)
    loaded = OmegaConf.load(Path("config/eval") / f"{config_name}.yaml")
    return OmegaConf.merge(default_config, loaded, OmegaConf.from_dotlist(overrides))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = (
        load_train_cfg(args.config_name, args.overrides)
        if args.mode == "train"
        else load_eval_cfg(args.config_name, args.overrides)
    )

    payload = {
        "mode": args.mode,
        "config_name": args.config_name,
        "logging_save_dir": str(getattr(cfg.logging, "save_dir", "")),
        "logging_run_name": str(getattr(cfg.logging, "wandb_run_name", "")),
        "logging_run_group": str(getattr(cfg.logging, "run_group", "")),
        "generation_server_port": int(getattr(cfg.generation, "server_port", 8005)),
        "huggingface_name": str(getattr(cfg.huggingface, "name", "")),
        "huggingface_push_to_hub": bool(
            getattr(getattr(cfg, "huggingface", None), "push_to_hub", False)
        ),
        "export_conversations_path": str(
            getattr(cfg, "export_conversations_path", "") or ""
        ),
        "export_metrics_path": str(getattr(cfg, "export_metrics_path", "") or ""),
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
