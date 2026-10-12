# 模态迁移（Modality Transfer）实验仓库

基于 [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 的图像到图像翻译代码，面向**多模态遥感/视觉数据**的模态迁移任务（例如光学 ↔ SAR、光学 ↔ 红外）。本仓库在上游 CycleGAN / pix2pix 实现之上，增加了针对模态迁移场景的训练/测试脚本示例、简易推理入口，以及基于 Flask 的异步推理服务端/客户端示例。

> **重要**：大型权重文件（`*.pth`、`checkpoints/`）**未上传**到本仓库（见根目录 `.gitignore`）。推理与服务端运行前，需自行准备对应实验名的生成器权重。

仓库地址：<https://github.com/MiaoHuai/test>（私有）

---

## 目录

1. [项目简介](#1-项目简介)
2. [仓库结构](#2-仓库结构)
3. [环境依赖](#3-环境依赖)
4. [数据准备](#4-数据准备)
5. [训练](#5-训练)
6. [测试与推理](#6-测试与推理)
7. [服务端 / 客户端](#7-服务端--客户端)
8. [可视化 Demo 网站](#8-可视化-demo-网站)
9. [Docker](#9-docker)
10. [注意事项](#10-注意事项)
11. [致谢与上游](#11-致谢与上游)

---

## 1. 项目简介

### 1.1 能力概览

| 能力 | 说明 |
|------|------|
| **pix2pix** | 成对（aligned）图像翻译，本仓库模态迁移示例主要使用该模型 |
| **CycleGAN** | 非成对（unaligned）图像翻译，保留上游完整训练/测试流程 |
| **自定义推理** | `infer.py`：按目录批量读图，经生成器输出并保存 |
| **HTTP 服务** | `server.py` + `run.py`：任务队列式异步推理（端口 `5001`） |
| **示例客户端** | `client.py` / `client_result.py`：提交任务与查询结果 |
| **可视化 Demo** | `demo/app.py`：中文网页，展示模式（无权重）+ 可选在线推理（端口 `7860`） |

### 1.2 本仓库侧重的模态迁移方向

从脚本注释、实验名与示例数据可以看出，本项目围绕以下 **pix2pix** 实验（权重需自行放置到 `checkpoints/<实验名>/`）：

| 简写 / 模态配置 | 实验目录名（`checkpoints` 下） | 含义（据命名） |
|-----------------|--------------------------------|----------------|
| `opt2sar` / `Vis-SAR` | `WHU-City_opt2sar_pix2pix` | 光学（可见光）→ SAR |
| `sar2opt` / `SAR-Vis` | `WHU-City_sar2opt_pix2pix` | SAR → 光学 |
| `opt2ir` / `Vis-Infrared` | `VEDAI_opt2ir_pix2pix` | 光学 → 红外 |
| `ir2opt` / `Infrared-Vis` | `VEDAI_ir2opt_pix2pix` | 红外 → 光学 |

部分脚本中数据集路径写作 `WHU-City`（如 `train.sh` / `test.sh`），另有工具脚本默认路径写作 `WHU-SEN-City`（如 `datasets/combine_A_and_B.py`、`copy_test_imgs.py`）。实际训练时请以你本地准备的数据目录为准。

仓库内提供少量示例测试图：

- `pytorch-CycleGAN-and-pix2pix/datasets/test/opt2sar/`（若干 PNG）
- `pytorch-CycleGAN-and-pix2pix/datasets/test/sar2opt/`（若干 PNG）

完整训练数据（WHU-City / VEDAI 等）**不在**本仓库中。

---

## 2. 仓库结构

```text
.
├── Dockerfile                          # 基于 PyTorch 镜像，安装依赖后启动 server.py（暴露 5001）
├── .gitignore                          # 忽略 checkpoints/、*.pth、results/ 等
├── README.md                           # 本说明（仓库根目录）
├── demo/                               # 可视化 Demo 网站（展示模式 + 可选在线推理）
│   ├── app.py                          # Flask 入口，默认端口 7860
│   ├── README.md                       # Demo 启动说明（中文）
│   ├── requirements.txt                # flask / pillow
│   ├── templates/ / static/            # 前端页面与样式
│   └── ...
└── pytorch-CycleGAN-and-pix2pix/       # 主代码（上游 + 本项目扩展）
    ├── train.py / test.py              # 通用训练 / 测试入口
    ├── train.sh / test.sh              # WHU-City opt2sar 的 pix2pix 示例命令
    ├── infer.py                        # 按目录批量推理（模态简写映射到实验名）
    ├── run.py                          # 服务端 Runner：按模态配置加载权重并批处理
    ├── server.py                       # Flask 异步任务服务（POST / 与 /result）
    ├── client.py / client_result.py    # 提交任务 / 查询任务状态示例
    ├── requirements.txt                # pip 依赖（核心）
    ├── environment.yml                 # conda 环境示例
    ├── options/                        # 命令行参数（base / train / test）
    ├── models/                         # cycle_gan、pix2pix、colorization、test 等
    ├── data/                           # aligned / unaligned / single 等 Dataset
    ├── datasets/                       # 数据工具脚本 + 少量 test 样例图
    ├── scripts/                        # 上游示例 train/test/download 脚本
    ├── util/                           # 可视化、HTML、图像池等
    └── docs/                           # 上游英文文档（overview、tips、qa 等）
```

工作目录请进入：

```bash
cd pytorch-CycleGAN-and-pix2pix
```

下文命令均默认在该目录下执行。

---

## 3. 环境依赖

### 3.1 系统要求

- Linux（推荐）或 macOS
- Python 3（上游与 `environment.yml` 示例为 **Python 3.8**）
- CPU 或 NVIDIA GPU + CUDA（训练强烈建议 GPU）

### 3.2 Conda（推荐，与 `environment.yml` 一致）

```bash
cd pytorch-CycleGAN-and-pix2pix
conda env create -f environment.yml
conda activate pytorch-CycleGAN-and-pix2pix
```

`environment.yml` 中锁定示例版本大致为：`pytorch=1.8.1`、`torchvision==0.9.1`、`Pillow`、`numpy`、`visdom`、`wandb`、`dominate` 等。

### 3.3 仅用 pip

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```text
torch>=1.4.0
torchvision>=0.5.0
dominate>=2.4.0
visdom>=0.1.8.8
wandb
```

### 3.4 推理 / 服务端额外依赖

以下包在扩展脚本中使用，但**未**写入 `requirements.txt`，使用对应功能前请自行安装：

| 包 | 用途 |
|----|------|
| `flask` | `server.py` HTTP 服务 |
| `requests` | `client.py` / `client_result.py` |
| `opencv-python`（`cv2`） | `infer.py`（部分注释代码）、`datasets/combine_A_and_B.py` 等 |
| `Pillow`、`tqdm` | 图像读写、`train.py` 进度条 |

示例：

```bash
pip install flask requests opencv-python pillow tqdm
```

### 3.5 可视化（可选）

训练时若 `--display_id > 0`，需启动 Visdom：

```bash
python -m visdom.server
# 默认 http://localhost:8097
```

不需要 Visdom 时可设 `--display_id -1`。

---

## 4. 数据准备

### 4.1 pix2pix（成对数据，本仓库模态迁移主路径）

pix2pix 默认 `--dataset_mode aligned`：每个样本为左右拼接的 **AB 图**（左为域 A，右为域 B），目录通常为：

```text
datasets/<数据集名>/
  ├── train/     # 训练用 AB 拼接图
  ├── val/       # 可选
  └── test/      # 测试用 AB 拼接图
```

若已有分离的 A、B 目录，可用上游工具拼接：

```bash
python datasets/combine_A_and_B.py \
  --fold_A /path/to/A \
  --fold_B /path/to/B \
  --fold_AB /path/to/AB_out
```

脚本默认参数中曾出现 `WHU-SEN-City/sar2opt` 路径，仅为历史默认值；请按本地数据修改。

本仓库示例训练脚本使用类似路径：

- `datasets/WHU-City/opt2sar`（见 `train.sh` / `test.sh`）

请自行下载或整理 WHU-SEN-City / WHU-City、VEDAI 等数据，并保证目录结构与 `--dataroot` 一致。

### 4.2 CycleGAN（非成对数据）

需要 `trainA` / `trainB`（及可选 `testA` / `testB`）等子目录。可用：

```bash
bash ./datasets/download_cyclegan_dataset.sh <数据集名>
```

### 4.3 仓库内自带样例

仅用于快速试跑推理路径（**不是**完整训练集）：

```text
datasets/test/opt2sar/*.png
datasets/test/sar2opt/*.png
```

`copy_test_imgs.py` 可从本地 `datasets/WHU-SEN-City/opt2sar/trainB` 随机抽取并缩放到 `datasets/test/sar2opt`，需本地已有完整数据。

---

## 5. 训练

### 5.1 通用命令

```bash
python train.py \
  --dataroot ./datasets/<你的数据> \
  --name <实验名> \
  --model pix2pix \          # 或 cycle_gan
  --direction AtoB \         # 或 BtoA，取决于 A/B 语义
  --gpu_ids 0
```

常用参数（详见 `options/base_options.py`、`options/train_options.py`）：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `--dataroot` | 数据集根目录 | `datasets/WHU-City/opt2sar` |
| `--name` | 实验名，决定 `checkpoints/<name>/` | `WHU-City_opt2sar_pix2pix` |
| `--model` | `pix2pix` / `cycle_gan` / `colorization` 等 | `pix2pix` |
| `--direction` | `AtoB` 或 `BtoA` | 见下方示例 |
| `--batch_size` | 批大小 | 视显存而定 |
| `--load_size` / `--crop_size` | 缩放与裁剪 | 如 286/256 或 600/512 |
| `--n_epochs` / `--n_epochs_decay` | 恒定 LR 与衰减阶段 epoch 数 | 默认 200 / 200 |
| `--continue_train` | 从最新 checkpoint 恢复 | 开关 |
| `--checkpoints_dir` | 权重保存根目录 | 默认 `./checkpoints` |

训练过程中会向 `checkpoints/<name>/` 写入网络权重（如 `latest_net_G.pth`、`latest_net_D.pth`）及 HTML 可视化（除非 `--no_html`）。

### 5.2 本仓库示例：`train.sh`（光学→SAR，pix2pix）

```bash
bash train.sh
# 等价于：
python train.py \
  --dataroot datasets/WHU-City/opt2sar \
  --name WHU-City_opt2sar_pix2pix_512 \
  --model pix2pix \
  --direction BtoA \
  --gpu_ids 0,1 \
  --batch_size 16 \
  --load_size 600 \
  --crop_size 512
```

说明：`--direction BtoA` 表示以拼接图中的 **B 为输入、A 为监督目标**（具体 A/B 语义取决于你制作数据时的左右顺序）。其他模态（sar2opt、opt2ir、ir2opt）可复制该脚本，修改 `--dataroot`、`--name`、`--direction` 与 GPU 设置。

### 5.3 上游示例脚本

```bash
bash scripts/train_pix2pix.sh
bash scripts/train_cyclegan.sh
```

更多技巧见 `docs/tips.md`、`docs/qa.md`。

---

## 6. 测试与推理

### 6.1 官方风格测试：`test.py`

加载 `checkpoints/<name>/` 中权重，对数据集跑推理，结果写入 `results/`（可用 `--results_dir` 指定），并生成 HTML。

示例（与仓库 `test.sh` 一致）：

```bash
bash test.sh
# 等价于：
python test.py \
  --dataroot datasets/WHU-City/opt2sar \
  --name WHU-City_opt2sar_pix2pix \
  --model pix2pix \
  --num_test 700 \
  --gpu_ids 7 \
  --load_size 256 \
  --crop_size 256 \
  --direction BtoA
```

注意：`test.py` 会强制 `batch_size=1`、`serial_batches`、`no_flip` 等；测试前请确保对应 `--name` 的权重已存在。

### 6.2 目录批量推理：`infer.py`

面向「输入目录中的 jpg/png → 生成器 → 输出目录」的简化流程，并将短名称映射到实验目录名：

| `--name` 传入 | 实际加载的实验名 |
|---------------|------------------|
| `opt2sar` | `WHU-City_opt2sar_pix2pix` |
| `sar2opt` | `WHU-City_sar2opt_pix2pix` |
| `ir2opt` | 意图映射到 `VEDAI_ir2opt_pix2pix`（见下方注意） |
| `opt2ir` | 意图映射到 `VEDAI_opt2ir_pix2pix`（见下方注意） |

示例（文件内注释）：

```bash
python infer.py \
  --dataroot datasets/test/opt2sar \
  --name opt2sar \
  --model pix2pix \
  --gpu_ids 0 \
  --results_dir results/opt2sar

python infer.py \
  --dataroot datasets/test/sar2opt \
  --name sar2opt \
  --model pix2pix \
  --gpu_ids 0 \
  --results_dir results/sar2opt
```

预处理：将图像 resize 到 256、归一化到 `[-1, 1]`，再经 `netG` 输出并反归一化保存。

> **代码注意**：`infer.py` 中 `ir2opt` / `opt2ir` 分支当前写成了比较运算符 `==` 而非赋值，短名可能不会正确映射；使用红外相关实验时，建议直接传入完整实验名，或先修正该赋值逻辑。

### 6.3 预训练权重下载（上游公开模型，非本仓库模态权重）

上游提供的 horse2zebra、facades 等公开模型：

```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```

**WHU-City / VEDAI 模态迁移权重不在本仓库，也不由上述脚本提供**，需自行训练或从内部存储拷贝到：

```text
checkpoints/WHU-City_opt2sar_pix2pix/latest_net_G.pth
checkpoints/WHU-City_sar2opt_pix2pix/latest_net_G.pth
checkpoints/VEDAI_opt2ir_pix2pix/latest_net_G.pth
checkpoints/VEDAI_ir2opt_pix2pix/latest_net_G.pth
```

（服务端 `run.py` 中部分路径写作容器内绝对路径 `/checkpoints/...`，部署时请与挂载方式一致。）

---

## 7. 服务端 / 客户端

### 7.1 启动服务

```bash
python server.py
# 监听 0.0.0.0:5001
```

`server.py` 会：

1. 后台线程初始化 `run.Runner`（期望加载四套模态权重）；
2. 使用队列串行执行任务；
3. 定期清理超过 12 小时的任务状态。

### 7.2 API

**提交任务** — `POST /`

请求体示例（见 `client.py`）：

```json
{
  "configs": [
    {
      "configId": 1,
      "filePaths": ["datasets/test/opt2sar/1.png"],
      "type": "Vis",
      "outputPath": "result/opt2sar",
      "modalityAugmentConfig": "Vis-SAR"
    }
  ]
}
```

`modalityAugmentConfig` 可选值（与 `run.py` 中 `set_name` 一致）：

- `Vis-SAR` → `WHU-City_opt2sar_pix2pix`
- `SAR-Vis` → `WHU-City_sar2opt_pix2pix`
- `Vis-Infrared` → `VEDAI_opt2ir_pix2pix`
- `Infrared-Vis` → `VEDAI_ir2opt_pix2pix`

成功时返回 `task_id`（`code: 200`）。

**查询结果** — `POST /result`

```json
{ "taskId": "<上一步返回的 id>" }
```

可用：

```bash
python client.py
python client_result.py <task_id>
```

### 7.3 服务端现状说明

`run.py` 为面向服务部署的草稿：`Runner.__init__` 中硬编码了四套 `/checkpoints/.../latest_net_G.pth` 路径，且部分成员在赋值前即调用 `load_state_dict`，`run_batch` 中亦混用了未完整初始化的配置。**部署前请按实际权重路径与模型创建逻辑审阅并修好 `run.py`**，再对外提供服务。本地单机推理更稳妥的路径是 `infer.py` 或 `test.py`。

---


## 8. 可视化 Demo 网站

仓库根目录 `demo/` 提供中文可视化页面，用于展示模态迁移效果：

- **展示模式**（默认，无需 `.pth`）：浏览 `datasets/test/` 中光学/SAR 样例对照，支持拖拽对比滑块。
- **在线推理**（可选）：当 `pytorch-CycleGAN-and-pix2pix/checkpoints/<实验名>/latest_net_G.pth` 存在时，可上传图片进行推理。

启动：

```bash
cd demo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
# 浏览器打开 http://127.0.0.1:7860
```

详细说明见 [`demo/README.md`](demo/README.md)。该 Demo **独立于** `server.py`（端口 5001 的异步任务 API）。

---

## 9. Docker

根目录 `Dockerfile` 概要：

- 基础镜像：`pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime`
- 将 `pytorch-CycleGAN-and-pix2pix` 拷贝到容器 `/workspace`
- 安装 OpenCV 相关系统库、`requirements.txt`，并尝试安装 pytorch3d 包
- 暴露端口 **5001**
- 默认命令：`python server.py`

构建与运行示例：

```bash
# 在仓库根目录
docker build -t modality-transfer:0321 .

docker run --gpus all -p 5001:5001 \
  -v /path/to/your/checkpoints:/checkpoints \
  -v /path/to/data:/workspace/datasets \
  modality-transfer:0321
```

请根据 `run.py` / `server.py` 实际读取的权重路径调整挂载点。镜像中的 pytorch3d 安装行与当前推理主路径无必然关系；若构建失败，可按需删改该层。

---

## 10. 注意事项

1. **权重未入库**  
   根目录 `.gitignore` 忽略 `**/checkpoints/`、`**/*.pth`、`**/results/` 等。GitHub 对大文件也有限制；请用网盘、对象存储或 Git LFS（若你方策略允许）单独管理权重。

2. **训练数据未入库**  
   完整 WHU-City / VEDAI 等数据集需自行准备；仓库仅含少量 `datasets/test/` 样例 PNG。

3. **GPU 编号**  
   示例脚本中的 `--gpu_ids 7` 等为原开发机配置，请改成你机器上可用的设备号；CPU 可设 `--gpu_ids -1`。

4. **实验名一致性**  
   `train.sh` 中实验名为 `WHU-City_opt2sar_pix2pix_512`，而 `test.sh` / `infer.py` 映射为 `WHU-City_opt2sar_pix2pix`。测试与推理时的 `--name` 必须与权重所在目录名一致。

5. **AppleDouble / macOS 垃圾文件**  
   从 macOS 拷贝时可能产生 `._*` 文件；已建议从版本库中清理并忽略。

6. **仅声明代码实际支持的内容**  
   CycleGAN、着色（colorization）等仍可按上游文档使用；本仓库自定义扩展主要围绕 **pix2pix 模态迁移 + 简易推理/服务**。

---

## 11. 致谢与上游

本仓库核心训练/测试框架来自：

- **Jun-Yan Zhu, Taesung Park 等**：[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- CycleGAN 论文：*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (ICCV 2017)
- pix2pix 论文：*Image-to-Image Translation with Conditional Adversarial Networks* (CVPR 2017)

上游许可证见 `pytorch-CycleGAN-and-pix2pix/LICENSE`。若在研究中使用，请按上游要求引用相关论文。

模态迁移相关扩展（`infer.py`、`server.py`、`run.py`、`train.sh` / `test.sh` 及示例数据等）为本仓库在上游基础上的工程化改动，用于光学 / SAR / 红外等模态互转实验与服务化尝试。

---

## 快速检查清单

- [ ] 已安装 PyTorch / torchvision 及第 3 节依赖  
- [ ] 已准备数据目录，或至少使用 `datasets/test/` 样例  
- [ ] 已将 `latest_net_G.pth` 放到对应 `checkpoints/<实验名>/`  
- [ ] 训练：`bash train.sh`（改路径与 GPU）  
- [ ] 测试：`bash test.sh` 或 `python infer.py ...`  
- [ ] （可选）服务：`python server.py`，再用 `client.py` 调用
- [ ] （可选）可视化 Demo：`cd demo && .venv/bin/python app.py` → http://127.0.0.1:7860  
