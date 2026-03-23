# 兼容说明

AWS 相关内容现在不再单独维护为主线。

请改看：

- [Vast Interruptible 部署手册](vast_interruptible_deploy_zh.md)
- [Vast Interruptible 恢复手册](vast_interruptible_recovery_zh.md)

如果你仍然在 AWS 上运行，可以直接沿用同样的命令体系，只要保证：

- GPU 正常
- 磁盘持久化
- `HF_TOKEN` / `huggingface-cli login` 已配置
