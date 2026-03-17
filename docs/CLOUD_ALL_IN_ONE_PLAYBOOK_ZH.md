# PedagogicalRL 云端唯一操作手册

这份文件的目标只有一个：

- 以后你上 AWS 或 GCP 时，只看这一个文件就够了

它会覆盖：

- 本地需要准备什么
- SSH 是什么，密钥从哪里来
- GitHub 和 Hugging Face 要准备什么
- W&B 怎么开
- 上机后每一步命令
- dry run 怎么跑
- full run 怎么跑
- Spot 中断后怎么续
- 怎么监控训练
- 怎么回收结果

---

## 0. 先记住 5 句话

1. 你不是在“部署网站”，你是在“远程登录一台 Linux 电脑，然后像本地一样跑命令”。
2. SSH 只是“登录远程服务器”的方式，不是网站，不是额外账号。
3. 模型第一次会下载到云服务器本地磁盘；只要磁盘还在，下次不用重下。
4. 这个项目已经支持从 checkpoint 自动续跑。
5. 你真正要避免的不是报错，而是“机器空转烧钱”。

---

## 1. 你现在已经有哪些东西

- 你的 GitHub 仓库：
  - [qcai0427/Teaching-LLM-replicate-AWS](https://github.com/qcai0427/Teaching-LLM-replicate-AWS)
- 这就是以后云服务器直接 clone 的仓库
- 最新云端基线提交：
  - `f26de9a`

这个仓库里已经包含：

- AWS / GCP 都可用的修复版代码
- FP8 dry run 配置
- no-FP8 dry run 配置
- FP8 full run 配置
- no-FP8 full run 配置
- 训练后评测配置

---

## 2. 名词翻译

### 2.1 SSH 是什么

SSH = 远程登录命令。

你以后在本地电脑终端里输入：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
```

意思是：

- 用你本地的密钥文件 `~/keys/aws-gpu.pem`
- 登录公网 IP 是 `<EC2_PUBLIC_IP>` 的那台云服务器
- 用户名是 `ubuntu`

### 2.2 `.pem` 是什么

它是 AWS 在创建 EC2 实例时给你的登录密钥文件。

它不是：

- GitHub 给你的
- Hugging Face 给你的
- 我生成的

它来自 AWS 控制台创建实例时的 `Key pair`。

### 2.3 W&B 是什么

W&B = Weights & Biases。

它是一个训练监控网站。开了之后你会更容易看到：

- 当前 step
- loss / reward 变化
- 训练是否还在走
- 哪个 run 对应哪次实验

它不是必须的，但强烈建议开。

---

## 3. 所有占位符是什么意思

你最害怕的是命令里有变量看不懂。下面统一解释。

### `<EC2_PUBLIC_IP>`

这表示云服务器的公网 IP 地址。

你去 AWS 控制台看实例详情时，会看到：

- `Public IPv4 address`

假设看到的是：

```text
3.145.88.120
```

那么命令里的：

```bash
ubuntu@<EC2_PUBLIC_IP>
```

就替换成：

```bash
ubuntu@3.145.88.120
```

### `~/keys/aws-gpu.pem`

这是你本地电脑上的密钥文件路径。

假设你把 AWS 下载下来的密钥文件放在：

```bash
~/keys/aws-gpu.pem
```

那就不改。

如果你实际放在：

```bash
~/Downloads/my-first-key.pem
```

那命令里就改成这个实际路径。

### `ubuntu`

这是云服务器用户名。

对于 Ubuntu 镜像，通常就是：

```bash
ubuntu
```

如果你选的是别的镜像，用户名可能不同。但 AWS 上大多数 Ubuntu AMI 都是 `ubuntu`。

### `~/work/Teaching-LLM-replicate-AWS`

这是云服务器上的项目目录。

意思是：

- 在云服务器的家目录下
- 有个 `work` 文件夹
- 里面 clone 了这个项目

### `pedagogy-aws`

这是 conda 环境名。

你以后看到：

```bash
conda activate pedagogy-aws
```

意思就是进入项目的 Python 环境。

---

## 4. 你本地电脑现在要准备什么

### 4.1 检查 SSH 客户端

你已经做过了，而且是好的：

```bash
ssh -V
```

看到 `OpenSSH_...` 就说明本地可以 SSH。

### 4.2 建一个放密钥的文件夹

在你本地电脑终端执行：

```bash
mkdir -p ~/keys
```

以后 AWS 下载下来的 `.pem` 文件放这里。

### 4.3 GitHub 需要准备什么

你已经准备好了：

- GitHub 账号
- 你的项目仓库

我们已经把代码推上去了，所以你之后云端直接 clone 就行。

### 4.4 Hugging Face 需要准备什么

你已经确认页面能访问，这很好。

你还需要：

1. 一个 Hugging Face token
2. 权限至少是 `Read`

### 4.5 W&B 需要准备什么

你需要：

1. 一个 [W&B](https://wandb.ai/) 账号
2. 一个 API key

拿 key 的方法：

1. 登录 W&B
2. 进入设置页面
3. 找到 API Keys
4. 复制你的 key

以后在云服务器上执行：

```bash
wandb login
```

然后把 key 粘进去。

或者你也可以直接设置环境变量：

```bash
export WANDB_API_KEY=你的key
```

---

## 5. 上云前的总体策略

不要一上来就 full run。

正确顺序：

1. 开机
2. 检查 GPU
3. clone 仓库
4. 建 conda 环境
5. 装依赖
6. 登录 Hugging Face
7. 登录 W&B
8. 预下载模型和数据
9. 单独测试 teacher / student / judge
10. 先跑 dry run
11. dry run 成功后，再跑 full run
12. full run 后评测
13. 打包结果
14. 下载回本地

---

## 6. 第一次创建 AWS 实例时你要拿到什么

你在 AWS 控制台创建 EC2 实例时，要创建一个新的 `Key pair`。

建议：

- 类型：`RSA`
- 格式：`.pem`

下载下来以后，把它放到你本地：

```bash
~/keys/aws-gpu.pem
```

然后在你本地终端执行：

```bash
chmod 400 ~/keys/aws-gpu.pem
```

这一步必须做。

---

## 7. 第一次 SSH 登录

在你本地电脑终端执行：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
```

例子：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@3.145.88.120
```

第一次连接时可能会问你：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入：

```text
yes
```

就行。

---

## 8. 登录云机后第一批命令

进入云机后，先做体检，不要急着 clone。

```bash
whoami
hostname
pwd
nvidia-smi
df -h
python3 --version
```

判断标准：

- `nvidia-smi` 能看到 GPU：继续
- `nvidia-smi` 不正常：停止，不要继续烧钱

---

## 9. clone 项目

在云机上执行：

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/qcai0427/Teaching-LLM-replicate-AWS.git
cd Teaching-LLM-replicate-AWS
git log --oneline -n 3
```

你应该能看到最新提交类似：

```text
f26de9a ...
```

---

## 10. 安装 conda

先试一下有没有 conda：

```bash
conda --version
```

如果没有，就执行：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
conda --version
```

---

## 11. 创建项目环境

```bash
conda create -n pedagogy-aws python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
python --version
```

---

## 12. 安装依赖

```bash
cd ~/work/Teaching-LLM-replicate-AWS
pip install -U pip setuptools wheel ninja packaging
pip install -r requirements.txt
```

如果 `flash-attn` 安装失败，再单独执行：

```bash
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

---

## 13. 验证 Python 栈

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import flash_attn; print('flash_attn ok')"
python -c "import vllm; print(vllm.__version__)"
```

---

## 14. 登录 Hugging Face

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

然后粘贴你的 HF token。

---

## 15. 开启 W&B

如果你要监控训练，强烈建议开。

### 15.1 安装并登录

```bash
pip install -U wandb
wandb login
```

然后粘贴你的 W&B API key。

建议：

- 不要把 API key 写进仓库文件
- 不要把 API key 写进公开文档
- 不要把 API key 直接写进长期保存的 shell 历史

### 15.2 这个项目里怎么打开 wandb

这个仓库的大部分训练配置默认已经是：

```yaml
logging:
  wandb: true
```

比如：

- [7b.yaml](/home/caiqi/phd/Codex/Teaching-LLM-replicate-AWS/config/train_rl/7b.yaml)
- [7b_aws_dryrun.yaml](/home/caiqi/phd/Codex/Teaching-LLM-replicate-AWS/config/train_rl/7b_aws_dryrun.yaml)
- [7b_nofp8.yaml](/home/caiqi/phd/Codex/Teaching-LLM-replicate-AWS/config/train_rl/7b_nofp8.yaml)

所以只要：

- 你已经 `wandb login`
- 配置里 `wandb: true`

训练就会自动往 W&B 记日志。

### 15.3 什么时候必须开，什么时候可以不开

建议你这样做：

- dry run 训练：可以开
- full run 训练：强烈建议开
- 评测：默认可以不开

原因：

- 训练时 W&B 最有价值，能帮你确认 run 还活着
- 评测时默认不开更安静，也能减少上传表格的麻烦

### 15.4 如果你不想开

可以在命令行里覆盖：

```bash
logging.wandb=false
```

例如：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun \
  logging.wandb=false
```

### 15.5 如果 W&B 出问题，会不会影响实验

默认情况下：

- 如果你已经登录成功，而且网络正常，W&B 只负责记录日志，不改变训练逻辑
- 如果你担心 W&B 成为不稳定因素，可以直接把它关掉，训练本身仍然可以继续

也就是说：

- W&B 不是训练的核心依赖
- 它更像是“监控面板”

最保守的策略是：

1. 先在便宜小机上单独测试一次 `wandb login`
2. 正式大机上如果 W&B 正常，就开着
3. 如果 W&B 报错或网络异常，就立刻改成 `logging.wandb=false`

### 15.6 W&B 的最小自检

在云机或便宜小机上执行：

```bash
pip install -U wandb
wandb login
wandb status
```

如果这三步都正常，再继续训练。

---

## 16. 先下载模型，不要直接训练

### 16.1 FP8 路线

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

### 16.2 no-FP8 路线

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

---

## 17. 下载数据集

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("rd211/Big-Math-RL-Verified-Filtered")
print(ds)
PY
```

然后看缓存大小：

```bash
du -sh ~/.cache/huggingface
df -h
```

---

## 18. 单独验证三个模型

### 18.1 Teacher

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("teacher ok", name)
PY
```

### 18.2 Student FP8

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student fp8 ok", name)
PY
```

### 18.3 Student no-FP8

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("student nofp8 ok", name)
PY
```

### 18.4 Judge

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype="auto")
print("judge ok", name)
PY
```

如果：

- FP8 student 过了：优先走 FP8 路线
- FP8 student 没过，但 no-FP8 student 过了：走 no-FP8 路线

---

## 19. 建日志目录

```bash
cd ~/work/Teaching-LLM-replicate-AWS
mkdir -p logs reports
```

---

## 20. 跑 dry run

### 20.1 FP8 dry run

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun \
  2>&1 | tee logs/aws_dryrun_train.log
```

### 20.2 no-FP8 dry run

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_aws_dryrun_nofp8 \
  2>&1 | tee logs/aws_dryrun_nofp8_train.log
```

---

## 21. 监控 dry run

另开一个本地终端，再次 SSH 登录：

```bash
ssh -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>
```

进入项目并激活环境：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
```

### 看 GPU

```bash
watch -n 1 nvidia-smi
```

### 看训练日志

FP8：

```bash
tail -f logs/aws_dryrun_train.log
```

no-FP8：

```bash
tail -f logs/aws_dryrun_nofp8_train.log
```

---

## 22. 跑 dry run 评测

### 22.1 FP8

```bash
python eval.py --config-name 7b_aws_dryrun 2>&1 | tee logs/aws_dryrun_eval.log
```

### 22.2 no-FP8

```bash
python eval.py --config-name 7b_aws_dryrun_nofp8 2>&1 | tee logs/aws_dryrun_nofp8_eval.log
```

---

## 23. 检查 dry run 是否成功

### 23.1 FP8

```bash
ls -lh checkpoints/7b-aws-dryrun
ls -lh reports/aws_dryrun_eval_conversations.json
ls -lh reports/aws_dryrun_eval_metrics.json
```

### 23.2 no-FP8

```bash
ls -lh checkpoints/7b-aws-dryrun-nofp8
ls -lh reports/aws_dryrun_nofp8_eval_conversations.json
ls -lh reports/aws_dryrun_nofp8_eval_metrics.json
```

---

## 24. 跑正式训练

### 24.1 论文风格 FP8 full run

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  2>&1 | tee logs/full_train_7b.log
```

### 24.2 no-FP8 full run

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8 \
  2>&1 | tee logs/full_train_7b_nofp8.log
```

---

## 25. 如果你用 Spot 实例

你提到一个很关键的问题：训练可能被中断很多次。

### 25.1 这个项目能续跑吗

能。

原因是训练脚本会自动查找最后一个 checkpoint，并从那里恢复。

也就是说：

- 你不需要手写 `resume_from_checkpoint`
- 只要原来的 checkpoint 目录还在
- 重新执行同一条训练命令，它就会自动续

### 25.2 续跑的前提

不是“代码有魔法”，而是：

- checkpoint 还在
- 磁盘没丢

所以非常重要：

- 训练目录必须放在**持久磁盘**上
- 不要把重要东西放到会随实例一起消失的临时盘

### 25.3 为 Spot 优化的正式训练命令

为了减少单次中断损失，建议更频繁保存 checkpoint。

FP8：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  logging.save_steps=2 \
  2>&1 | tee logs/full_train_7b_spot.log
```

no-FP8：

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8 \
  logging.save_steps=2 \
  2>&1 | tee logs/full_train_7b_nofp8_spot.log
```

### 25.4 Spot 中断后如何续跑

重新 SSH 上去，进入项目目录，再执行**同一条命令**。

FP8：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b \
  logging.save_steps=2 \
  2>&1 | tee -a logs/full_train_7b_spot.log
```

no-FP8：

```bash
cd ~/work/Teaching-LLM-replicate-AWS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pedagogy-aws
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b_nofp8 \
  logging.save_steps=2 \
  2>&1 | tee -a logs/full_train_7b_nofp8_spot.log
```

### 25.5 AWS 和 GCP 上 Spot 的存储提醒

我只给你讲最保守的原则，不赌云平台默认行为：

- 一定确认你的训练目录在持久磁盘上
- 一定确认实例停止或抢占后磁盘不会被自动删除
- 不要把“能续跑”建立在临时磁盘上

只要磁盘在，代码就能续。

---

## 26. 正式训练完成后评测

### 26.1 FP8 训练产物评测

```bash
python eval.py --config-name 7b_local 2>&1 | tee logs/7b_eval.log
```

### 26.2 no-FP8 训练产物评测

```bash
python eval.py --config-name 7b_local_nofp8 2>&1 | tee logs/7b_nofp8_eval.log
```

---

## 27. 结果回收

### 27.1 看有哪些结果

```bash
ls -lh checkpoints
ls -lh reports
ls -lh logs
```

### 27.2 打包结果

FP8：

```bash
tar -czf results_7b.tar.gz checkpoints/7b reports/7b_eval_conversations.json reports/7b_eval_metrics.json logs/full_train_7b.log logs/7b_eval.log
```

no-FP8：

```bash
tar -czf results_7b_nofp8.tar.gz checkpoints/7b-nofp8 reports/7b_nofp8_eval_conversations.json reports/7b_nofp8_eval_metrics.json logs/full_train_7b_nofp8.log logs/7b_nofp8_eval.log
```

### 27.3 下载回本地

在你**本地电脑**终端执行，不是在云机里执行。

FP8：

```bash
scp -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>:~/work/Teaching-LLM-replicate-AWS/results_7b.tar.gz .
```

例子：

```bash
scp -i ~/keys/aws-gpu.pem ubuntu@3.145.88.120:~/work/Teaching-LLM-replicate-AWS/results_7b.tar.gz .
```

no-FP8：

```bash
scp -i ~/keys/aws-gpu.pem ubuntu@<EC2_PUBLIC_IP>:~/work/Teaching-LLM-replicate-AWS/results_7b_nofp8.tar.gz .
```

---

## 28. 我到底能看到哪些进度

### 28.1 你能看到的

- `eval.py` 有 tqdm 进度条
- `classroom.py` 的 rollout 阶段有 tqdm
- `tail -f logs/...` 能看到训练在继续
- `watch -n 1 nvidia-smi` 能看到 GPU 忙不忙
- 开了 W&B 后，网页上能看到 run 的曲线和 step 变化

### 28.2 你看不到的

这个项目没有专门做一个“超级准确 ETA 剩余时间预测器”。

所以：

- 能看到进度
- 能看到 step
- 能看到日志
- 但剩余时间通常只能粗估

最现实的办法：

1. 观察最近几个 step 大概多久
2. 用剩余 step 数乘一下

---

## 29. 离开前最后检查

### 看有没有训练还在跑

```bash
ps -ef | grep -E "train_rl.py|vllm_server.py|uvicorn" | grep -v grep
```

### 如果已经不跑了

去云平台控制台：

- `Stop instance`

不是：

- `Terminate instance`

---

## 30. 你最容易出错的地方

1. 把本地命令和云机命令搞混
2. 不知道 `<EC2_PUBLIC_IP>` 要替换成真实 IP
3. 忘记 `chmod 400` `.pem`
4. 还没验证模型，就直接 full run
5. Spot 中断后没确认 checkpoint 是否还在
6. 训练结束忘记 stop 实例

---

## 31. 你到时候真的只要照着做的最短路线

### 第一天

1. SSH 登录
2. 看 `nvidia-smi`
3. clone 项目
4. 装 conda
5. 装依赖
6. HF login
7. W&B login
8. 下载模型
9. 下载数据
10. stop 实例

### 第二天

1. 再次 SSH 登录
2. 激活环境
3. 单独验证 teacher / student / judge
4. dry run
5. dry run eval
6. stop 实例

### 如果你先租的是便宜小服务器

那你只做这些，不要尝试 full run：

1. SSH 登录
2. clone 项目
3. conda 环境创建
4. `pip install -r requirements.txt`
5. `huggingface-cli login`
6. `wandb login`
7. `wandb status`
8. 可选：下载数据集

便宜小机的目标是熟悉流程，不是验证 4 卡训练。

### 第三天

1. 再次 SSH 登录
2. full run
3. full eval
4. 打包结果
5. `scp` 下载回本地
6. stop 实例

---

## 32. 最后一句

你不是不会远程部署，你只是第一次做。

这件事在本质上就是：

- 登录一台远程 Linux 机器
- 进入项目目录
- 像本地一样敲命令

如果你严格按这个文件执行，你不需要现场自己发明流程。
