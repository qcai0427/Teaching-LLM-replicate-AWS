# 云服务器第一天操作手册

## 0. 先记住这三句话

- GPU 云服务器从你启动开始就会按时间计费。
- 但是当你把实例 `stop` 掉以后，计算资源费用会停止，只剩磁盘等存储费用继续计费。
- Hugging Face 模型第一次会在云机上下载到本地缓存；只要你不删磁盘，这些模型下次重启还在，不需要重新下载。

## 1. 你的目标不是“立刻开训”

第一天的目标只有三个：

1. 进入云机
2. 确认 GPU、驱动、CUDA、网络、磁盘正常
3. 让项目完成一次小规模 dry run

如果这三件事做到了，你就已经非常成功了。

## 2. 先决定你要走哪条路

### 路线 A：AWS EC2 g7e.24xlarge

- 4 x RTX PRO 6000 Blackwell Server Edition 96GB
- 更大显存
- 更适合先验证 FP8 / vLLM 路径是否稳定

### 路线 B：GCP a2-ultragpu-4g

- 4 x A100 80GB
- 更接近论文环境

如果你老师让你自己选，而且你最担心踩环境坑，我建议先选 AWS。

## 3. 开机前准备

你本地电脑上先准备：

- 一个放 SSH 密钥的文件夹
- 你的 GitHub 账号
- Hugging Face 账号
- W&B 账号（如果后面要记录训练）

## 4. 在 AWS/GCP 控制台里创建实例时的原则

### 4.1 最重要的原则

- 尽量选择带 GPU 驱动和 CUDA 的镜像
- 选够大的磁盘
- 只开放 SSH
- 先不要开放任何公网额外端口

### 4.2 磁盘建议

- 最少 `500GB`
- 更稳妥是 `1TB`

原因：

- 模型会下载到本地缓存
- 数据集也会缓存
- 训练会写 checkpoint
- 日志和中间文件也会占空间

## 5. 如果你用 AWS，最推荐的创建方式

### 5.1 建议镜像

优先选择：

- Ubuntu 22.04
- 或者 AWS 的 Deep Learning AMI / 带 NVIDIA 驱动的 GPU 镜像

### 5.2 安全组

只开：

- `22/tcp`

并且 Source 设成：

- 你的当前公网 IP

不要一上来开 `0.0.0.0/0`

### 5.3 SSH 密钥

创建一个新的 key pair，然后把 `.pem` 下载到你本地电脑。

假设文件名叫：

`aws-gpu.pem`

把它放到：

```bash
~/keys/aws-gpu.pem
```

## 6. 从你本地电脑第一次 SSH 登录

### 6.1 先改权限

在你本地电脑终端执行：

```bash
chmod 400 ~/keys/aws-gpu.pem
```

### 6.2 登录命令

如果镜像是 Ubuntu，通常用户名是 `ubuntu`：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
```

如果镜像不是 Ubuntu，用户名可能不同，但大多数 Ubuntu AMI 都是 `ubuntu`。

## 7. 一登录进去先干什么

不要先 clone，不要先装环境，不要先开训练。

先执行下面这些命令：

```bash
whoami
hostname
pwd
```

```bash
nvidia-smi
```

```bash
df -h
```

```bash
python3 --version
```

## 8. 如何判断现在可不可以继续

### 正常情况

`nvidia-smi` 能看到 4 张 GPU，并且有显存信息。

### 不正常情况

如果 `nvidia-smi` 报错，或者根本找不到命令：

- 先不要继续
- 先不要装项目依赖
- 先不要下载模型

因为这说明驱动 / CUDA / AMI 本身没有准备好。

## 9. 建项目目录

在云机上执行：

```bash
mkdir -p ~/work
cd ~/work
```

## 10. clone 仓库

```bash
git clone <你的仓库地址或本地同步后的仓库地址> Teaching-LLM-replicate-AWS
cd Teaching-LLM-replicate-AWS
```

如果你是直接把本地目录推到 GitHub 了，就 clone 你的版本。

## 11. 安装 conda

如果镜像里已经有 conda，可以先试：

```bash
conda --version
```

如果没有，再装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
conda --version
```

## 12. 创建项目环境

```bash
conda create -n pedagogy-aws python=3.11 -y
conda activate pedagogy-aws
```

确认：

```bash
which python
python --version
```

## 13. 安装基础依赖

```bash
pip install -U pip setuptools wheel ninja packaging
pip install -r requirements.txt
```

## 14. 如果 flash-attn 安装失败

再单独装：

```bash
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 15. 安装完依赖后的第一次验证

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

```bash
python -c "import flash_attn; print('flash_attn ok')"
```

```bash
python -c "import vllm; print(vllm.__version__)"
```

## 16. 登录 Hugging Face

如果模型需要认证，或者你想提高下载稳定性：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

按提示粘贴你的 token。

## 17. 为什么模型会下载

这个项目会在第一次运行时从 Hugging Face 下载模型，并缓存到本地磁盘。

缓存通常在：

```bash
~/.cache/huggingface
```

这不是坏事，反而是你省钱的关键：

- 第一次下载慢，但之后就不需要重复下载
- 只要你 stop 实例而不删磁盘，缓存还在

## 18. 先手动预下载模型，不要直接训练

这是省钱关键步骤。

注意：

- 仓库配置里当前写的是 `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
- Hugging Face 现在会把这个旧路径重定向到 `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`
- 也就是说，项目代码里保留旧路径通常仍能工作，但你手动检查模型页面和权限时，应当同时认识这两个名字实际上指向同一个模型页面

在项目目录里执行：

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

然后预下载数据集：

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("rd211/Big-Math-RL-Verified-Filtered")
print(ds)
PY
```

## 19. 预下载完成后立刻检查缓存大小

```bash
du -sh ~/.cache/huggingface
df -h
```

## 20. 这时如果你今天不继续干了，怎么办

如果你今天只是完成环境和模型缓存：

- **直接 stop 实例**
- 不要 terminate

因为：

- stop 后计算费用停止
- 但磁盘和模型缓存会保留
- 下次重新启动就能接着用

## 21. 第二次上机先干什么

先重新 SSH 登录，然后：

```bash
nvidia-smi
cd ~/work/Teaching-LLM-replicate-AWS
conda activate pedagogy-aws
```

再检查缓存是否还在：

```bash
du -sh ~/.cache/huggingface
```

## 22. 正式开始 dry run 前的最后验证

### 22.1 teacher 可加载测试

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("teacher ok", name)
PY
```

### 22.2 student 可加载测试

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student ok", name)
PY
```

### 22.3 judge 可加载测试

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("judge ok", name)
PY
```

如果这三步里任何一步失败，先停，不要继续烧 GPU 时间。

## 22.4 如果 student 的 FP8 模型加载失败怎么办

不要现场改代码，直接切换到仓库里已经准备好的无 FP8 配置：

- 训练配置：`config/train_rl/7b_aws_dryrun_nofp8.yaml`
- 评测配置：`config/eval/7b_aws_dryrun_nofp8.yaml`

这套配置只替换 student 为 `meta-llama/Llama-3.1-8B-Instruct`，其余 dry run 结构保持一致。

它的作用不是更忠实复现论文，而是把风险隔离开：

- 如果 FP8 路径正常，就跑 `7b_aws_dryrun`
- 如果 FP8 路径不正常，就先跑 `7b_aws_dryrun_nofp8`

这样你可以先验证整条训练链路，再决定是否继续排查 FP8 兼容性。

## 23. 启动 dry run 训练

在项目目录：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
conda activate pedagogy-aws
mkdir -p logs reports
```

启动训练并落盘日志：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun \
  2>&1 | tee logs/aws_dryrun_train.log
```

如果 student FP8 路径失败，就改成：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun_nofp8 \
  2>&1 | tee logs/aws_dryrun_nofp8_train.log
```

## 24. 另开一个终端监控训练

新开一个 SSH 终端，执行：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
```

进入项目：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
conda activate pedagogy-aws
```

监控 GPU：

```bash
watch -n 1 nvidia-smi
```

或者看日志：

```bash
tail -f logs/aws_dryrun_train.log
```

## 25. 训练结束后跑评测

```bash
python eval.py --config-name 7b_aws_dryrun 2>&1 | tee logs/aws_dryrun_eval.log
```

如果你跑的是无 FP8 配置，就执行：

```bash
python eval.py --config-name 7b_aws_dryrun_nofp8 2>&1 | tee logs/aws_dryrun_nofp8_eval.log
```

## 26. dry run 成功的标准

你只需要看到这些就算成功：

- 训练完成至少 2 个 step
- 生成 checkpoint
- 评测正常结束
- 导出了下面两个文件：

```bash
ls -lh reports/aws_dryrun_eval_conversations.json
ls -lh reports/aws_dryrun_eval_metrics.json
```

## 27. dry run 成功后，才允许正式训练

正式训练命令：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  2>&1 | tee logs/aws_full_train.log
```

## 28. 每次离开电脑前必须做的事

先检查是不是还在训练：

```bash
ps -ef | grep -E "train_rl.py|vllm_server.py|uvicorn" | grep -v grep
```

如果还在训练，就不要 stop。

如果不在训练，并且今天结束了：

- 回到 AWS 控制台
- 选择实例
- 点击 `Stop instance`

## 29. 最省钱的工作习惯

- 不要一开机就开始下载一堆不必要的东西
- 每次只做一个明确目标
- 当天目标完成就 stop
- 不要把“看文档、思考、整理笔记”的时间浪费在 GPU 开机状态上

## 30. 你最应该警惕的三个坑

### 坑 1：`nvidia-smi` 不正常还继续往下做

这是最浪费钱的做法。

### 坑 2：student FP8 模型加载失败还硬上训练

先解决模型兼容性，再谈训练。

### 坑 3：训练结束忘了 stop 实例

这是新手最常见也最贵的错误。

## 31. 今天如果你只能记住一件事

你第一次上云，不需要证明自己会“正式训练”。

你只需要完成下面这条链：

- 登陆成功
- GPU 正常
- 环境装好
- 模型缓存好
- dry run 成功
- 记得 stop 实例

这就已经是一次非常成功的云端实验启动。
