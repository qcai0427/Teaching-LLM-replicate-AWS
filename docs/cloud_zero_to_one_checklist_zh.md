# 云端从零到一执行清单

这份清单只回答一件事：第一次上云时，你应该依次输入什么命令。

## 1. 本地电脑上

假设你的密钥文件在：

```bash
~/keys/aws-gpu.pem
```

先改权限：

```bash
chmod 400 ~/keys/aws-gpu.pem
```

登录云机：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
```

## 2. 云机登录后，先不要训练

先检查机器状态：

```bash
whoami
hostname
pwd
nvidia-smi
df -h
python3 --version
```

如果 `nvidia-smi` 不正常，停下，不要继续。

## 3. 建工作目录并 clone 仓库

```bash
mkdir -p ~/work
cd ~/work
git clone <你的仓库地址> Teaching-LLM-replicate-AWS
cd Teaching-LLM-replicate-AWS
```

## 4. 安装 conda

如果系统里已经有 conda：

```bash
conda --version
```

如果没有：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
conda --version
```

## 5. 创建项目环境

```bash
conda create -n pedagogy-aws python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
python --version
```

## 6. 安装依赖

```bash
pip install -U pip setuptools wheel ninja packaging
pip install -r requirements.txt
```

如果 `flash-attn` 没装好，再单独执行：

```bash
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 7. 验证 Python 栈

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import flash_attn; print('flash_attn ok')"
python -c "import vllm; print(vllm.__version__)"
```

## 8. 登录 Hugging Face

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

## 9. 预下载模型

注意：

- 仓库配置里写的是 `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
- Hugging Face 当前规范页面会跳到 `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`
- 两者当前指向同一个模型页面

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

## 10. 预下载数据集

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("rd211/Big-Math-RL-Verified-Filtered")
print(ds)
PY
```

## 11. 检查缓存和磁盘

```bash
du -sh ~/.cache/huggingface
df -h
```

如果你当天只做到这里，就去云平台控制台点 `Stop instance`。

## 12. 第二次上机后验证三个模型都能加载

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

Student:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student ok", name)
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

只要其中一个失败，就先停。

如果 student 的 FP8 模型失败，不要改源码，直接改用：

- `config/train_rl/7b_aws_dryrun_nofp8.yaml`
- `config/eval/7b_aws_dryrun_nofp8.yaml`

## 13. 启动 dry run

```bash
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
mkdir -p logs reports
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun \
  2>&1 | tee logs/aws_dryrun_train.log
```

如果 student FP8 不通：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun_nofp8 \
  2>&1 | tee logs/aws_dryrun_nofp8_train.log
```

## 14. 另开一个终端监控

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
watch -n 1 nvidia-smi
```

或者：

```bash
tail -f ~/work/Teaching-LLM-replicate-AWS/logs/aws_dryrun_train.log
```

## 15. 训练后评测

```bash
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
python eval.py --config-name 7b_aws_dryrun 2>&1 | tee logs/aws_dryrun_eval.log
```

如果你跑的是无 FP8 版本：

```bash
python eval.py --config-name 7b_aws_dryrun_nofp8 2>&1 | tee logs/aws_dryrun_nofp8_eval.log
```

## 16. 成功判据

```bash
ls -lh checkpoints/7b-aws-dryrun
ls -lh reports/aws_dryrun_eval_conversations.json
ls -lh reports/aws_dryrun_eval_metrics.json
```

## 17. 离开前最后检查

```bash
ps -ef | grep -E "train_rl.py|vllm_server.py|uvicorn" | grep -v grep
```

如果没有训练在跑，就去控制台点 `Stop instance`，不要让机器空转。
