# Teaching-LLM Replicate AWS

这个仓库现在以 `Vast.ai interruptible` 为主线，目标是把训练、评测、恢复、W&B 监控和 Hugging Face 回收串成一条可重复执行的链路。

## 主入口

- 环境安装：`./scripts/install_vast_env.sh`
- 仅训练，可中断恢复：`./run_vast_training.sh`
- 训练 -> 评测 -> 上传 Hugging Face：`./run_vast_job.sh`

## 目录约定

- `checkpoints/`：checkpoint 和最终导出模型
- `logs/`：训练、评测、上传、vLLM 日志
- `reports/`：评测 conversations / metrics
- `.runtime/`：PID 文件和运行摘要

## 推荐配置

- `config/train_rl/7b_vast_dryrun.yaml`
- `config/eval/7b_vast_dryrun.yaml`
- `config/train_rl/7b_vast_dryrun_nofp8.yaml`
- `config/eval/7b_vast_dryrun_nofp8.yaml`
- `config/train_rl/7b.yaml`
- `config/train_rl/7b_nofp8.yaml`

## 快速示例

```bash
./run_vast_job.sh \
  --job-name vast-dryrun \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun \
  --eval-config-name 7b_vast_dryrun \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_MODEL_REPO
```

如果你只想先验证训练恢复，而不上传：

```bash
./run_vast_training.sh \
  --job-name vast-train-check \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun
```

## 详细文档

- [Vast Interruptible 全流程手册](docs/vast_interruptible_deploy_zh.md)

旧的 AWS/GCP 文档仍保留为兼容入口，但不再是主线。
