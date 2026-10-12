# 模态迁移可视化 Demo

面向仓库 `MiaoHuai/test` 的中文可视化演示页：支持 **展示模式**（无权重也可运行）与 **在线推理**（检测到 `checkpoints` 下对应 `.pth` 时启用）。

## 功能

| 模式 | 条件 | 能力 |
|------|------|------|
| **展示模式** | 默认；无需权重 | 浏览 `datasets/test/opt2sar` 与 `sar2opt` 样例对照；拖拽对比滑块 |
| **在线推理** | 存在 `latest_net_G.pth` | 上传图片，按所选方向调用 pix2pix 生成器 |

支持的迁移方向：

- `SAR → 光学`（`WHU-City_sar2opt_pix2pix`）
- `光学 → SAR`（`WHU-City_opt2sar_pix2pix`）
- `红外 → 光学`（`VEDAI_ir2opt_pix2pix`，需权重；仓库暂无红外样例）
- `光学 → 红外`（`VEDAI_opt2ir_pix2pix`，需权重；仓库暂无红外样例）

> **说明**：展示模式中的光学/SAR 对照按编号对齐，仅用于界面演示，**不是**同一场景的真实推理结果。页面不会声称「无权重时模型已在推理」。

## 快速启动

在仓库根目录执行：

```bash
cd demo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

浏览器打开：**http://127.0.0.1:7860**

可选环境变量：

- `DEMO_HOST`（默认 `0.0.0.0`）
- `DEMO_PORT`（默认 `7860`）

在线推理额外需要在 `pytorch-CycleGAN-and-pix2pix` 环境中安装 PyTorch / torchvision（与主项目一致），并将权重放到：

```text
pytorch-CycleGAN-and-pix2pix/checkpoints/<实验名>/latest_net_G.pth
```

## 目录结构

```text
demo/
├── app.py              # Flask 入口
├── requirements.txt    # flask / pillow
├── README.md           # 本说明
├── templates/index.html
├── static/css/style.css
├── static/js/main.js
├── uploads/            # 运行时上传（gitignore）
└── results/            # 运行时结果（gitignore）
```

样例图片仍位于主代码目录：

`pytorch-CycleGAN-and-pix2pix/datasets/test/{opt2sar,sar2opt}/`

## 与现有 server.py 的关系

本 Demo **独立**于 `pytorch-CycleGAN-and-pix2pix/server.py`（后者为异步任务队列 API，端口 5001）。可视化站点使用 `demo/app.py`（端口 7860），在检测到权重时复用同一套实验命名与生成器加载思路（对齐 `infer.py`）。
