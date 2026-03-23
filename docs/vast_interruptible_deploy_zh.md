# Vast Interruptible 全流程手册

这是一份从头到尾的完整手册。你只需要看这一份，就能完成：

- 云端部署
- 环境安装
- Hugging Face 登录
- W&B 登录和监控
- dry run
- 正式训练
- 完整评测
- Hugging Face 自动上传
- 中断恢复
- 最后确认收工

## 1. 你现在这套仓库是怎么工作的

你先建立一个持久磁盘上的工作目录，然后训练过程把关键状态写到四个地方：

- `checkpoints/`：训练 checkpoint 和最终导出的模型目录
- `logs/`：训练、评测、上传、vLLM 的完整日志
- `reports/`：评测结果文件
- `.runtime/`：运行摘要、PID 文件、最近一次运行状态

这意味着 interruptible 实例被打断以后，只要磁盘还在，你就能继续。

仓库现在有三个主入口：

- `./scripts/install_vast_env.sh`：安装环境
- `./run_vast_training.sh`：只跑训练，适合测试恢复
- `./run_vast_job.sh`：训练 -> 评测 -> 上传 Hugging Face

## 2. 在 Vast.ai 上创建实例时的原则

- GPU：优先 4 卡，显存越大越稳
- 磁盘：至少 `500GB`，更建议 `1TB`
- 镜像：优先 Ubuntu + CUDA
- 网络：允许 SSH
- 最重要的一点：确认磁盘是持久化的，实例停掉后数据不会一起消失

## 3. 第一次 SSH 上云后先体检

```bash
whoami
hostname
pwd
nvidia-smi
df -h
python3 --version
```

如果 `nvidia-smi` 不正常，先停，不要继续。

## 4. clone 仓库

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/qcai0427/Teaching-LLM-replicate-AWS.git
cd Teaching-LLM-replicate-AWS
git log --oneline -n 3
```

## 5. 创建 conda 环境

如果机器没有 conda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

创建项目环境：

```bash
conda create -n pedagogy-vast python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-vast
python --version
```

## 6. 安装依赖

默认安装：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
./scripts/install_vast_env.sh
```

如果你的镜像需要指定 PyTorch wheel 源，例如 CUDA 12.4：

```bash
./scripts/install_vast_env.sh --torch-index-url https://download.pytorch.org/whl/cu124
```

如果你要强制升级 `ninja`：

```bash
./scripts/install_vast_env.sh --with-ninja-upgrade
```

## 7. 安装后最小验证

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import vllm; print(vllm.__version__)"
python -c "import transformers, datasets, accelerate; print('python stack ok')"
```

## 8. 登录 Hugging Face

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

如果你更习惯环境变量：

```bash
export HF_TOKEN=你的_hf_token
```

检查是否登录成功：

```bash
huggingface-cli whoami
```

## 9. 登录 W&B

如果你有 W&B key，建议在正式 dry run 前就完成登录。

```bash
pip install -U wandb
wandb login
```

或者直接设置环境变量：

```bash
export WANDB_API_KEY=你的_wandb_key
```

登录后检查状态：

```bash
wandb status
```

说明：

- 训练配置默认会把训练信息打到 W&B
- 评测默认不开 W&B
- 如果某次你不想开 W&B，可以在运行命令最后追加：

```bash
-- \
logging.wandb=false
```

## 10. 预下载模型

### 10.1 FP8 路线

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
]
for model in models:
    print("downloading", model)
    print(snapshot_download(repo_id=model))
PY
```

### 10.2 no-FP8 路线

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
]
for model in models:
    print("downloading", model)
    print(snapshot_download(repo_id=model))
PY
```

## 11. 预下载数据

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("rd211/Big-Math-RL-Verified-Filtered")
print(ds)
PY
```

检查缓存和磁盘：

```bash
du -sh ~/.cache/huggingface
df -h
```

## 12. 单独验证模型能否加载

Teacher：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-7B-Instruct"
AutoTokenizer.from_pretrained(name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("teacher ok")
PY
```

Student FP8：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
AutoTokenizer.from_pretrained(name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student fp8 ok")
PY
```

Student no-FP8：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "meta-llama/Llama-3.1-8B-Instruct"
AutoTokenizer.from_pretrained(name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student nofp8 ok")
PY
```

Judge：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
AutoTokenizer.from_pretrained(name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("judge ok")
PY
```

如果 FP8 student 失败，就改走：

- `config/train_rl/7b_vast_dryrun_nofp8.yaml`
- `config/train_rl/7b_nofp8.yaml`

## 13. dry run：只测试训练恢复

```bash
mkdir -p checkpoints logs reports .runtime
./run_vast_training.sh \
  --job-name vast-dryrun-train \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun
```

no-FP8：

```bash
./run_vast_training.sh \
  --job-name vast-dryrun-train-nofp8 \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun_nofp8
```

这个命令的目的只是验证：

- vLLM 能否启动
- 训练能否前进
- checkpoint 能否产生
- 中断后能否恢复

## 14. dry run：训练 + 评测 + 自动上传 Hugging Face

FP8：

```bash
./run_vast_job.sh \
  --job-name vast-dryrun \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun \
  --eval-config-name 7b_vast_dryrun \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_MODEL_REPO
```

no-FP8：

```bash
./run_vast_job.sh \
  --job-name vast-dryrun-nofp8 \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun_nofp8 \
  --eval-config-name 7b_vast_dryrun_nofp8 \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_MODEL_REPO_NOFP8
```

如果你想先保留本地结果、不上传：

```bash
./run_vast_job.sh \
  --job-name vast-dryrun-local \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_vast_dryrun \
  --eval-config-name 7b_vast_dryrun \
  --skip-upload
```

## 15. 完整训练 + 完整评测 + 自动上传

FP8 正式版：

```bash
./run_vast_job.sh \
  --job-name vast-full-7b \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b \
  --eval-config-name 7b_vast_dryrun \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_FULL_MODEL_REPO
```

no-FP8 正式版：

```bash
./run_vast_job.sh \
  --job-name vast-full-7b-nofp8 \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b_nofp8 \
  --eval-config-name 7b_vast_dryrun_nofp8 \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_FULL_MODEL_REPO_NOFP8
```

如果你要临时覆盖超参，把 override 放在最后：

```bash
./run_vast_job.sh \
  --job-name vast-full-custom \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b \
  --eval-config-name 7b_vast_dryrun \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_FULL_MODEL_REPO \
  -- \
  logging.save_steps=2 \
  train.max_steps=20
```

## 16. 如果你只想做完整评测

当训练已经完成、`checkpoints/.../model` 已经存在时，可以单独评测。

FP8：

```bash
python eval.py \
  --config-name 7b_vast_dryrun \
  teacher_model.model_name_or_path=checkpoints/7b/model \
  export_conversations_path=reports/full_eval_conversations.json \
  export_metrics_path=reports/full_eval_metrics.json
```

no-FP8：

```bash
python eval.py \
  --config-name 7b_vast_dryrun_nofp8 \
  teacher_model.model_name_or_path=checkpoints/7b-nofp8/model \
  export_conversations_path=reports/full_eval_nofp8_conversations.json \
  export_metrics_path=reports/full_eval_nofp8_metrics.json
```

## 17. 运行时怎么监控

GPU：

```bash
watch -n 1 nvidia-smi
```

训练日志：

```bash
tail -f logs/vast-dryrun_train_full.log
```

评测日志：

```bash
tail -f logs/vast-dryrun_eval_full.log
```

上传日志：

```bash
tail -f logs/vast-dryrun_upload_full.log
```

vLLM 日志：

```bash
tail -f logs/vast-dryrun_vllm_server.log
```

W&B：

- 打开你的 project
- 看 step 是否增长
- 看 loss / reward 是否刷新
- 如果命令行和 W&B 都长时间不动，再排查问题

## 18. 怎么判断 dry run 成功

- vLLM server 启动成功
- 训练至少完成 2 个 step
- `checkpoints/7b-vast-dryrun/` 里出现 checkpoint 和 `model/`
- `reports/vast-dryrun_eval_conversations.json` 存在
- `reports/vast-dryrun_eval_metrics.json` 存在
- 如果启用了上传，Hugging Face repo 中出现模型和 `runs/vast-dryrun/` 下的评测产物

## 19. 怎么判断完整训练、完整评测、上传都收工了

先确认没有训练进程：

```bash
ps -ef | grep -E "train_rl.py|vllm_server.py|uvicorn" | grep -v grep
```

再确认关键目录：

```bash
ls -lah checkpoints
ls -lah reports
ls -lah logs
ls -lah .runtime
```

查看运行摘要：

```bash
cat .runtime/vast-full-7b_run_summary.json
```

如果一切正常，你应该同时看到：

- `checkpoints/<run>/model/` 存在
- 对应的 `reports/*.json` 存在
- `.runtime/*_run_summary.json` 里的 `status` 是 `uploaded`，或者至少是 `evaluated`
- Hugging Face repo 已经更新
- W&B 上 run 不再继续增长

## 20. interruptible 打断后怎么恢复

重新 SSH 回实例：

```bash
whoami
hostname
pwd
nvidia-smi
df -h
```

重新进入仓库和环境：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-vast
```

确认持久产物还在：

```bash
ls -lah checkpoints
ls -lah reports
ls -lah logs
ls -lah .runtime
du -sh ~/.cache/huggingface
```

看上次运行摘要：

```bash
cat .runtime/vast-dryrun_train_status.json
cat .runtime/vast-dryrun_run_summary.json
```

然后直接重跑同一条命令，不要改 `save_dir`。

只恢复训练：

```bash
./run_vast_training.sh \
  --job-name vast-full-7b \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b
```

恢复整条 job：

```bash
./run_vast_job.sh \
  --job-name vast-full-7b \
  --train-config-file config/deepspeed/zero3_4GPU.yaml \
  --train-config-name 7b \
  --eval-config-name 7b_vast_dryrun \
  --hf-repo-id YOUR_HF_USERNAME/YOUR_FULL_MODEL_REPO
```

如果你之前用了 override，也必须带回同样的 override。

## 21. 怎么判断恢复是真的从上次接着跑

训练输出里应该看到：

- `Resume checkpoint: checkpoints/.../checkpoint-N`
- `Last checkpoint: checkpoints/.../checkpoint-N`

目录里应该看到更大的 checkpoint：

```bash
find checkpoints/7b -maxdepth 1 -type d -name 'checkpoint-*' | sort
```

如果有更大的 `checkpoint-*` 出现，说明恢复成功。

## 22. 训练和评测成功了，但 Hugging Face 上传失败怎么办

看上传日志：

```bash
tail -f logs/vast-full-7b_upload_full.log
```

确认 HF 登录状态：

```bash
huggingface-cli whoami
```

然后手动补传：

```bash
python3 scripts/upload_run_to_hub.py \
  --repo-id YOUR_HF_USERNAME/YOUR_FULL_MODEL_REPO \
  --model-dir checkpoints/7b/model \
  --artifact-prefix runs/vast-full-7b \
  --metrics-file reports/vast-full-7b_eval_metrics.json \
  --conversations-file reports/vast-full-7b_eval_conversations.json \
  --summary-file .runtime/vast-full-7b_run_summary.json
```

## 23. 如果 vLLM PID 留脏了

```bash
./stop_vllm_server.sh
```

然后再重跑训练命令。

## 24. 今天准备收工时做什么

先确认训练进程已经结束：

```bash
ps -ef | grep -E "train_rl.py|vllm_server.py|uvicorn" | grep -v grep
```

再确认关键产物已经落盘：

```bash
ls -lah checkpoints
ls -lah reports
ls -lah logs
ls -lah .runtime
```

如果需要，再到 Hugging Face 和 W&B 页面各看一眼：

- Hugging Face 上模型 repo 是否已经更新
- W&B run 是否已经停止刷新

确认无误后，再去 Vast 控制台停止实例。不要删除磁盘。
