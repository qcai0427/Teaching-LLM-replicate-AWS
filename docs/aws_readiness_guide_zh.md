# PedagogicalRL AWS 运行指南

## 1. 目标

这个仓库不是为本地 A5500 单卡烟雾测试准备的，而是为后续迁移到 AWS 多卡环境准备的干净工作区。

它的原则是：

- 保留对 AWS 也有意义的修复
- 不带入本地 A5500 特供的小模型 smoke 配置
- 先做 dry run，再做正式训练

## 2. 相比 upstream 保留了哪些修改

### 必要修复

- `src/classroom.py`
  - 将 OpenRouter / Gemini provider 改为延迟导入
  - 目的：避免未使用这些 provider 时，由缺包触发顶层导入失败

- `src/vllm/data_parallel_vllm.py`
  - cleanup 更稳健
  - 目的：减少 vLLM worker 在退出阶段抛异常

- `train_rl.py` / `train_sft.py`
  - `attn_implementation` 由配置驱动，不再硬编码
  - 目的：让 AWS 机器和其他机器切换 attention 后端更容易

- `eval.py`
  - 增加更稳的 cleanup
  - 增加 tqdm 进度显示
  - 增加本地 conversation / metrics 导出

## 3. 这份仓库里新增了什么

- `config/train_rl/7b_aws_dryrun.yaml`
  - AWS 上正式训练前的小规模 dry run 配置

- `config/eval/7b_aws_dryrun.yaml`
  - dry run 训练后的小规模评测配置

- `config/train_rl/7b_aws_dryrun_nofp8.yaml`
  - student 不用 FP8 的备用 dry run 配置

- `config/eval/7b_aws_dryrun_nofp8.yaml`
  - 无 FP8 dry run 对应评测配置

- `config/train_rl/7b_nofp8.yaml`
  - 如果 student 的 FP8 路径在云端不稳定，可用于正式训练的备用配置

## 4. 不要做的事

- 不要把本地 A5500 smoke 配置当成 AWS 正式训练配置
- 不要第一次上 AWS 就直接长时间 full run
- 不要跳过 student / judge / vLLM server 的单独加载检查

## 5. 建议的 AWS 机器

如果你们老师后面给的是接近论文设置的机器，优先假设：

- 4 x A100 80GB
- Ubuntu
- CUDA 12.x
- conda 环境

## 6. 首次上机后的推荐步骤

### Step 1. clone 仓库

```bash
git clone <your_repo_or_local_copy> Teaching-LLM-replicate-AWS
cd Teaching-LLM-replicate-AWS
```

### Step 2. 建环境

```bash
conda create -n pedagogy-aws python=3.11 -y
conda activate pedagogy-aws
pip install -U pip setuptools wheel ninja packaging
pip install -r requirements.txt
```

如果 `flash-attn` 编译失败，再单独安装：

```bash
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 7. 正式训练前必须先验证的三件事

### 7.1 teacher 能否加载

先确认 `Qwen/Qwen2.5-7B-Instruct` 可正常被 transformers 和 vLLM 加载。

### 7.2 student 能否加载

重点检查：

- `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
- vLLM 0.8.3
- 当前 GPU 架构是否稳定支持该路径

这一步必须单独确认，因为它是整个项目最大的环境风险点之一。

### 7.3 judge 能否加载

确认：

- `Qwen/Qwen2.5-14B-Instruct-AWQ`
- vLLM 下可正常启动和生成

## 8. Dry run 推荐顺序

### 8.1 先启动 server

```bash
python vllm_server.py --config-name 7b_aws_dryrun
```

另开一个终端检查：

```bash
curl http://localhost:8005/docs
```

### 8.2 跑训练 dry run

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun
```

如果 student FP8 路径失败，就改跑：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun_nofp8
```

### 8.3 跑评测 dry run

```bash
python eval.py --config-name 7b_aws_dryrun
```

如果你跑的是无 FP8 版本，就执行：

```bash
python eval.py --config-name 7b_aws_dryrun_nofp8
```

## 9. Dry run 成功的判据

- vLLM server 能成功启动
- 训练能完成至少 2 个 step
- 能正常写出 checkpoint
- eval 能正常结束
- `reports/aws_dryrun_eval_conversations.json` 被导出
- `reports/aws_dryrun_eval_metrics.json` 被导出

## 10. Dry run 成功后怎么切回正式配置

如果 dry run 没问题，再切回作者原始的正式配置：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b
```

如果你已经确认 student 的 FP8 路径在当前云机上不稳定，但主链路和其他模型都正常，那么正式训练时可以改用：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8
```

这不是论文原样复现，但它能把“方法是否能在你的云环境上稳定训练”与“某个 FP8 checkpoint 的兼容性”拆开。

## 11. 你向导师汇报时建议这样说

- 本地已经验证过代码主链路能跑通
- 当前 AWS 仓库保留了必要修复，但没有带入本地单卡特供配置
- 上云后先跑 dry run，再跑正式训练
- 如果 dry run 通过，就说明代码和环境已经具备正式训练条件

## 12. 目前最需要警惕的问题

- student FP8 模型的 vLLM 兼容性
- deepspeed / flash-attn / CUDA 版本匹配
- vLLM server 与训练脚本的并行稳定性

## 13. 推荐日志观察方式

训练时建议分别看：

```bash
watch -n 1 nvidia-smi
```

```bash
tail -f logs/train.log
```

如果你要自己做更规范的落盘，建议把训练和评测都显式重定向到日志文件。
