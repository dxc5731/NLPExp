# 服务器操作手册

> 这个包里已经有：训练代码、数据（train/dev）、运行脚本、自检脚本。
> 你不需要在这台服务器上装 DSH，也不需要联网装别的东西（除了 setup 那一步）。

## 0. 先把包传上去

在**你自己的 Windows 电脑**上，PowerShell 里执行（把 `user@host` 换成你的服务器）：

```powershell
cd D:\agent-chat\dsh-deepseek\nlp\handoff
scp nlp_train_pkg.tar.gz user@host:~/
```

如果包还没生成，先在本地跑一次 `python code\scripts\build_pkg.py`。

## 1. 解压并准备环境（约 10—20 分钟，主要是在下 2.4G 的 PyTorch）

```bash
ssh user@host
tar xzf ~/nlp_train_pkg.tar.gz -C ~/
cd ~/nlp_train_pkg
bash setup.sh
```

**预期最后几行**：

```
   batch 形状 (8, ...)
   loss 0.xxxx  前向+反向 xxx ms

全部自检通过，可以开始训练。
```

看到这句就说明环境、数据、模型、GPU 全部正常。

**如果自检没过**，把从报错开始到最后 20 行发回来，不用自己猜。

## 2. 跑训练

```bash
bash run_train.sh 3e-5
```

约 20—30 分钟（取决于卡）。跑完会打印最后的指标。

想连续跑两组可以接着来：

```bash
bash run_train.sh 5e-5
```

### 有闲卡就并行跑（推荐）

服务器空卡多的时候，一次把三组学习率同时跑完，墙钟时间和跑一组一样：

```bash
bash run_parallel.sh 2e-5 3e-5 5e-5
```

每组自动占一张卡（GPU 0/1/2），日志各写一份 `code/run_<lr>.log`。
**三组必须在同一台机器上跑**，否则机器不同，结果不可比——本地那个 3e-5 的基线算预实验，
正式对比表用服务器这三组。

内存参考：一个进程约占 3GB 内存和 3—4GB 显存，三组并行要 10GB 内存以上。
不够就少跑几组，或者 `bash run_parallel.sh 2e-5 5e-5`。

## 3. 把结果拿回来

```bash
bash collect.sh
```

会生成一个几 KB 的 `results_xxx.tar.gz`，里面是每轮的指标、日志和配置。
在本地执行 `scp user@host:~/nlp_train_pkg/results_xxx.tar.gz .` 拉回来给我。

**如果我要看最优权重**（比如要在服务器上跑 dev 推理），用 `bash collect.sh yes`，
文件会变成 400MB 左右。

## 4. 中断了怎么办

`run_train.sh` 是普通前台进程，Ctrl+C 就停。已经跑完的轮次日志和权重都在
`code/runs/<run_id>/`，不会丢。重新跑一次就是新的 run_id，不会覆盖旧的。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `bash: setup.sh: /bin/bash^M: bad interpreter` | 文件被转成了 Windows 换行 | `sed -i 's/\r$//' setup.sh run_train.sh collect.sh` |
| `setup.sh` 卡在 pip 下载 | 网络慢 | 正常，2.4G 要几分钟；实在不行换 `--index-url https://download.pytorch.org/whl/cu121` |
| 自检报 `cuda_available=False` | 装成了 CPU 版 torch | 重跑 setup 的第 2 步 |
| 自检报显存不足 | 卡被别人占了 | `nvidia-smi` 看一下，换个空卡（`CUDA_VISIBLE_DEVICES=1 bash run_train.sh 3e-5`） |
| 下载模型失败 | 服务器访问不了 HuggingFace | setup 会自动换 hf-mirror；还不行就要离线拷模型 |
