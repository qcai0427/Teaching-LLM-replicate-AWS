# 云端逐命令操作指南

这份文件只做一件事：给出从登录云机到训练、续跑、评测、回收结果的逐条命令。

## 1. 第一次登录云机后

```bash
whoami
hostname
pwd
nvidia-smi
df -h
python3 --version
```

如果 `nvidia-smi` 不正常，停下，不要继续。

## 2. clone 仓库

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/qcai0427/Teaching-LLM-replicate-AWS.git
cd Teaching-LLM-replicate-AWS
git log --oneline -n 3
```

## 3. 安装 conda

如果机器里没有 conda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## 4. 创建环境

```bash
conda create -n pedagogy-aws python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
python --version
```

## 5. 安装依赖

```bash
cd ~/work/Teaching-LLM-replicate-AWS
pip install -U pip setuptools wheel ninja packaging
pip install -r requirements.txt
```

如果 `flash-attn` 失败：

```bash
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 6. 验证 Python 栈

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import flash_attn; print('flash_attn ok')"
python -c "import vllm; print(vllm.__version__)"
```

## 7. 登录 Hugging Face

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

## 8. 预下载模型

FP8 路线：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
]
for m in models:
    print(f"Downloading {m} ...")
    path = snapshot_download(repo_id=m)
    print(f"Saved to {path}")
PY
```

无 FP8 路线：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
]
for m in models:
    print(f"Downloading {m} ...")
    path = snapshot_download(repo_id=m)
    print(f"Saved to {path}")
PY
```

## 9. 预下载数据集

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("rd211/Big-Math-RL-Verified-Filtered")
print(ds)
PY
```

## 10. 检查缓存

```bash
du -sh ~/.cache/huggingface
df -h
```

## 11. 单独验证三个模型

Teacher:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("teacher ok", name)
PY
```

Student FP8:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student fp8 ok", name)
PY
```

Student no FP8:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student nofp8 ok", name)
PY
```

Judge:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("judge ok", name)
PY
```

## 12. 建日志目录

```bash
cd ~/work/Teaching-LLM-replicate-AWS
mkdir -p logs reports
```

## 13. 启动 dry run

FP8 版：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun \
  2>&1 | tee logs/aws_dryrun_train.log
```

无 FP8 版：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun_nofp8 \
  2>&1 | tee logs/aws_dryrun_nofp8_train.log
```

## 14. 监控 dry run

新开一个 SSH 终端：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
```

看 GPU：

```bash
watch -n 1 nvidia-smi
```

看日志：

```bash
tail -f logs/aws_dryrun_train.log
```

或：

```bash
tail -f logs/aws_dryrun_nofp8_train.log
```

## 15. dry run 评测

FP8 版：

```bash
python eval.py --config-name 7b_aws_dryrun 2>&1 | tee logs/aws_dryrun_eval.log
```

无 FP8 版：

```bash
python eval.py --config-name 7b_aws_dryrun_nofp8 2>&1 | tee logs/aws_dryrun_nofp8_eval.log
```

## 16. 检查 dry run 结果

FP8 版：

```bash
ls -lh checkpoints/7b-aws-dryrun
ls -lh reports/aws_dryrun_eval_conversations.json
ls -lh reports/aws_dryrun_eval_metrics.json
```

无 FP8 版：

```bash
ls -lh checkpoints/7b-aws-dryrun-nofp8
ls -lh reports/aws_dryrun_nofp8_eval_conversations.json
ls -lh reports/aws_dryrun_nofp8_eval_metrics.json
```

## 17. 启动正式训练

论文风格 FP8 版：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  2>&1 | tee logs/full_train_7b.log
```

无 FP8 版：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8 \
  2>&1 | tee logs/full_train_7b_nofp8.log
```

## 18. Spot 实例更稳妥的正式训练命令

为了减少被打断时损失的步数，建议把 `save_steps` 降低。

FP8 版：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  logging.save_steps=2 \
  2>&1 | tee logs/full_train_7b_spot.log
```

无 FP8 版：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8 \
  logging.save_steps=2 \
  2>&1 | tee logs/full_train_7b_nofp8_spot.log
```

## 19. Spot 中断后续跑

这个项目会自动检查 `save_dir` 下最近的 checkpoint，并从那里恢复。

所以你不需要写额外的 resume 参数，只要：

- 还在同一个工作目录
- 还保留着原来的 `checkpoints/...`
- 再次执行**同一条训练命令**

例如，FP8 版续跑：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  logging.save_steps=2 \
  2>&1 | tee -a logs/full_train_7b_spot.log
```

无 FP8 版续跑：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8 \
  logging.save_steps=2 \
  2>&1 | tee -a logs/full_train_7b_nofp8_spot.log
```

## 20. 正式训练完成后评测

FP8 版本地模型：

```bash
python eval.py --config-name 7b_local 2>&1 | tee logs/7b_eval.log
```

无 FP8 版本地模型：

```bash
python eval.py --config-name 7b_local_nofp8 2>&1 | tee logs/7b_nofp8_eval.log
```

## 21. 回收结果

查看主要产物：

```bash
ls -lh checkpoints
ls -lh reports
ls -lh logs
```

打包结果：

```bash
tar -czf results_7b.tar.gz checkpoints/7b reports/7b_eval_conversations.json reports/7b_eval_metrics.json logs/full_train_7b.log logs/7b_eval.log
```

无 FP8 版：

```bash
tar -czf results_7b_nofp8.tar.gz checkpoints/7b-nofp8 reports/7b_nofp8_eval_conversations.json reports/7b_nofp8_eval_metrics.json logs/full_train_7b_nofp8.log logs/7b_nofp8_eval.log
```

## 22. 下载结果到本地电脑

在你本地电脑终端执行：

```bash
scp -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>:~/work/Teaching-LLM-replicate-AWS/results_7b.tar.gz .
```

或：

```bash
scp -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>:~/work/Teaching-LLM-replicate-AWS/results_7b_nofp8.tar.gz .
```

## 23. 查看当前是否还在训练

```bash
ps -ef | grep -E "train_rl.py|vllm_server.py|uvicorn" | grep -v grep
```

## 24. 你能看到什么进度

- `eval.py` 有 tqdm 进度条
- `src/classroom.py` 的 rollout 阶段有 tqdm 进度条
- `watch -n 1 nvidia-smi` 可以看 GPU 是否还在工作
- `tail -f logs/...` 可以看训练是否还在推进

## 25. 预计剩余时间怎么判断

这个项目没有为完整 RL 训练额外实现一个非常准确的“剩余时间”估计器。

最现实的做法是：

1. 先看已经完成了多少 step
2. 看最近几个 step 的时间间隔
3. 用剩余步数乘以平均 step 时间做粗估

如果你把 `wandb: true` 打开，云端网页里也会更容易观察 step 速度变化。

## 26. Spot 实例必须注意的事

- 如果云平台把实例直接删掉，而且磁盘也删掉，那 checkpoint 就没了
- 所以一定要使用会保留磁盘的方案
- 训练目录和 Hugging Face 缓存必须放在持久磁盘上
- Spot 续跑的前提不是“脚本会魔法恢复”，而是“你的 checkpoint 还在”
