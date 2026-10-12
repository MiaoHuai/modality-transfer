#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模态迁移可视化 Demo（展示模式 + 可选在线推理）。

运行::

    cd demo
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python app.py

默认 http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image
from werkzeug.utils import secure_filename

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
CYCLEGAN_DIR = REPO_ROOT / "pytorch-CycleGAN-and-pix2pix"
SAMPLE_ROOT = CYCLEGAN_DIR / "datasets" / "test"
CHECKPOINTS_DIR = CYCLEGAN_DIR / "checkpoints"
UPLOAD_DIR = DEMO_DIR / "uploads"
RESULT_DIR = DEMO_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

DIRECTIONS = {
    "sar2opt": {
        "id": "sar2opt",
        "label": "SAR → 光学",
        "short": "SAR→Vis",
        "desc": "将 SAR（合成孔径雷达）图像迁移为光学可见光风格",
        "input_label": "输入（SAR）",
        "output_label": "输出（光学）",
        "modality": "SAR-Vis",
        "experiment": "WHU-City_sar2opt_pix2pix",
        "sample_input_dir": "sar2opt",
        "sample_output_dir": "opt2sar",
        "showcase": True,
        "dataset": "WHU-City",
    },
    "opt2sar": {
        "id": "opt2sar",
        "label": "光学 → SAR",
        "short": "Vis→SAR",
        "desc": "将光学可见光图像迁移为 SAR 风格",
        "input_label": "输入（光学）",
        "output_label": "输出（SAR）",
        "modality": "Vis-SAR",
        "experiment": "WHU-City_opt2sar_pix2pix",
        "sample_input_dir": "opt2sar",
        "sample_output_dir": "sar2opt",
        "showcase": True,
        "dataset": "WHU-City",
    },
    "ir2opt": {
        "id": "ir2opt",
        "label": "红外 → 光学",
        "short": "IR→Vis",
        "desc": "将红外图像迁移为光学可见光风格（需 VEDAI 权重）",
        "input_label": "输入（红外）",
        "output_label": "输出（光学）",
        "modality": "Infrared-Vis",
        "experiment": "VEDAI_ir2opt_pix2pix",
        "sample_input_dir": None,
        "sample_output_dir": None,
        "showcase": False,
        "dataset": "VEDAI",
    },
    "opt2ir": {
        "id": "opt2ir",
        "label": "光学 → 红外",
        "short": "Vis→IR",
        "desc": "将光学可见光图像迁移为红外风格（需 VEDAI 权重）",
        "input_label": "输入（光学）",
        "output_label": "输出（红外）",
        "modality": "Vis-Infrared",
        "experiment": "VEDAI_opt2ir_pix2pix",
        "sample_input_dir": None,
        "sample_output_dir": None,
        "showcase": False,
        "dataset": "VEDAI",
    },
}

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

app = Flask(
    __name__,
    static_folder=str(DEMO_DIR / "static"),
    template_folder=str(DEMO_DIR / "templates"),
)

_model_cache = {}


def checkpoint_path(direction_id):
    meta = DIRECTIONS.get(direction_id)
    if not meta:
        return None
    p = CHECKPOINTS_DIR / meta["experiment"] / "latest_net_G.pth"
    return p if p.is_file() else None


def live_status():
    status = {}
    for did, meta in DIRECTIONS.items():
        cp = checkpoint_path(did)
        status[did] = {
            "live": cp is not None,
            "checkpoint": str(cp) if cp else None,
            "experiment": meta["experiment"],
        }
    return status


def list_sample_ids(folder_name):
    if not folder_name:
        return []
    d = SAMPLE_ROOT / folder_name
    if not d.is_dir():
        return []
    ids = []
    for f in sorted(d.iterdir()):
        if f.name.startswith("._"):
            continue
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            ids.append(f.stem)
    return ids


def showcase_pairs(direction_id):
    meta = DIRECTIONS[direction_id]
    if not meta["showcase"]:
        return []
    in_ids = list_sample_ids(meta["sample_input_dir"])
    out_ids = set(list_sample_ids(meta["sample_output_dir"]))
    pairs = []
    for sid in in_ids:
        if sid not in out_ids:
            continue
        pairs.append(
            {
                "id": sid,
                "title": "样例 {}".format(sid),
                "caption": (
                    "【展示模式】仓库内光学/SAR 样例对照（编号 {}），"
                    "用于界面演示；并非同一场景的真实推理结果。"
                    "放置 .pth 权重后可启用在线推理。"
                ).format(sid),
                "before_url": url_for(
                    "sample_image",
                    folder=meta["sample_input_dir"],
                    filename="{}.png".format(sid),
                ),
                "after_url": url_for(
                    "sample_image",
                    folder=meta["sample_output_dir"],
                    filename="{}.png".format(sid),
                ),
                "before_label": meta["input_label"],
                "after_label": meta["output_label"],
            }
        )
    return pairs


def try_load_generator(direction_id):
    """尝试加载 pix2pix 生成器；失败或不存在权重时返回 None。"""
    if direction_id in _model_cache:
        return _model_cache[direction_id]

    cp = checkpoint_path(direction_id)
    if cp is None:
        _model_cache[direction_id] = None
        return None

    try:
        import torch
        from torchvision import transforms
    except ImportError:
        app.logger.warning("PyTorch 未安装，无法启用在线推理")
        _model_cache[direction_id] = None
        return None

    cg = str(CYCLEGAN_DIR)
    if cg not in sys.path:
        sys.path.insert(0, cg)

    old_argv = sys.argv
    try:
        sys.argv = [
            "demo_infer",
            "--dataroot", str(SAMPLE_ROOT),
            "--name", DIRECTIONS[direction_id]["experiment"],
            "--model", "pix2pix",
            "--gpu_ids", "-1",
            "--checkpoints_dir", str(CHECKPOINTS_DIR),
            "--direction", "AtoB",
            "--dataset_mode", "single",
            "--load_size", "256",
            "--crop_size", "256",
            "--preprocess", "resize_and_crop",
            "--no_flip",
            "--num_threads", "0",
            "--batch_size", "1",
            "--serial_batches",
            "--eval",
        ]
        from options.test_options import TestOptions
        from models import create_model

        opt = TestOptions().parse()
        opt.num_threads = 0
        opt.batch_size = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.display_id = -1
        opt.isTrain = False
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        bundle = {"model": model, "transform": transform, "torch": torch}
        _model_cache[direction_id] = bundle
        return bundle
    except Exception as e:
        app.logger.exception("加载权重失败 %s: %s", direction_id, e)
        _model_cache[direction_id] = None
        return None
    finally:
        sys.argv = old_argv


def run_inference(direction_id, image):
    import numpy as np

    bundle = try_load_generator(direction_id)
    if bundle is None:
        raise RuntimeError("该方向暂无可用权重，无法在线推理")

    model = bundle["model"]
    transform = bundle["transform"]
    torch = bundle["torch"]

    img = image.convert("RGB")
    with torch.no_grad():
        tensor = transform(img)[None]
        device = next(model.netG.parameters()).device
        tensor = tensor.to(device)
        out = model.netG(tensor)[0].detach().cpu().float().numpy()

    out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
    out = out.astype("uint8")
    return Image.fromarray(out)


@app.route("/")
def index():
    status = live_status()
    any_live = any(v["live"] for v in status.values())
    return render_template(
        "index.html",
        directions=list(DIRECTIONS.values()),
        live_status=status,
        any_live=any_live,
        mode_label="在线推理可用" if any_live else "展示模式（无权重）",
    )


@app.route("/api/status")
def api_status():
    status = live_status()
    return jsonify({
        "ok": True,
        "any_live": any(v["live"] for v in status.values()),
        "directions": {
            did: {
                **DIRECTIONS[did],
                "live": status[did]["live"],
                "checkpoint": status[did]["checkpoint"],
                "showcase_count": len(showcase_pairs(did)),
            }
            for did in DIRECTIONS
        },
    })


@app.route("/api/samples/<direction_id>")
def api_samples(direction_id):
    if direction_id not in DIRECTIONS:
        return jsonify({"ok": False, "error": "未知方向"}), 404
    pairs = showcase_pairs(direction_id)
    meta = DIRECTIONS[direction_id]
    live = checkpoint_path(direction_id) is not None
    if live:
        note = "已检测到该方向权重，可上传图片进行在线推理；也可浏览下方样例。"
    elif meta["showcase"]:
        note = (
            "展示模式：以下为仓库样例图库中的光学/SAR 对照（编号对齐），"
            "用于界面演示；并非带权重的真实推理输出。"
        )
    else:
        note = "该方向暂无仓库样例，也未检测到权重文件。"
    return jsonify({
        "ok": True,
        "direction": direction_id,
        "label": meta["label"],
        "live": live,
        "showcase": meta["showcase"],
        "note": note,
        "pairs": pairs,
    })


@app.route("/samples/<folder>/<path:filename>")
def sample_image(folder, filename):
    if folder not in {"opt2sar", "sar2opt"}:
        return "Not found", 404
    safe = secure_filename(filename)
    return send_from_directory(SAMPLE_ROOT / folder, safe)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    safe = secure_filename(filename)
    return send_from_directory(UPLOAD_DIR, safe)


@app.route("/results/<path:filename>")
def result_file(filename):
    safe = secure_filename(filename)
    return send_from_directory(RESULT_DIR, safe)


@app.route("/api/infer", methods=["POST"])
def api_infer():
    direction_id = request.form.get("direction", "")
    if direction_id not in DIRECTIONS:
        return jsonify({"ok": False, "error": "未知迁移方向"}), 400

    if checkpoint_path(direction_id) is None:
        exp = DIRECTIONS[direction_id]["experiment"]
        return jsonify({
            "ok": False,
            "live": False,
            "error": (
                "未找到权重文件，无法在线推理。"
                "请将 latest_net_G.pth 放到 "
                "pytorch-CycleGAN-and-pix2pix/checkpoints/{}/ "
                "后重启服务。当前请使用展示模式浏览样例。"
            ).format(exp),
        }), 503

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "请上传图片"}), 400
    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "空文件"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": "不支持的格式: {}".format(ext)}), 400

    uid = uuid.uuid4().hex[:12]
    in_name = "{}_in{}".format(uid, ext)
    out_name = "{}_out.png".format(uid)
    in_path = UPLOAD_DIR / in_name
    out_path = RESULT_DIR / out_name
    f.save(in_path)

    try:
        img = Image.open(in_path)
        result = run_inference(direction_id, img)
        result.save(out_path)
    except Exception as e:
        return jsonify({"ok": False, "error": "推理失败: {}".format(e)}), 500

    return jsonify({
        "ok": True,
        "live": True,
        "direction": direction_id,
        "before_url": url_for("uploaded_file", filename=in_name),
        "after_url": url_for("result_file", filename=out_name),
        "before_label": DIRECTIONS[direction_id]["input_label"],
        "after_label": DIRECTIONS[direction_id]["output_label"],
    })


def main():
    host = os.environ.get("DEMO_HOST", "0.0.0.0")
    port = int(os.environ.get("DEMO_PORT", "7860"))
    print("=" * 60)
    print("模态迁移可视化 Demo")
    print("  打开: http://127.0.0.1:{}".format(port))
    status = live_status()
    live_dirs = [d for d, s in status.items() if s["live"]]
    if live_dirs:
        print("  在线推理方向: {}".format(", ".join(live_dirs)))
    else:
        print("  当前为展示模式（未检测到 checkpoints/*.pth）")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
