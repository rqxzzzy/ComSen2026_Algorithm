import os
import time
import shutil

import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

ROOT = r"D:\ComSen2026\task4"
WEIGHTS = os.path.join(ROOT, "weights")
OUT_DIR = os.path.join(ROOT, "outputs")
DATA_YAML = os.path.join(ROOT, "datasets", "drone_bird", "data.yaml")
VIDEO = os.path.join(ROOT, "data", "example", "example.mp4")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

WARMUP = 10
IMGSZ = 640


def file_mb(p):
    return os.path.getsize(p) / 1048576 if os.path.exists(p) else 0.0


def get_sample_frames(n=30):
    import cv2
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def bench(model, frames, imgsz=IMGSZ, device=0):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(WARMUP):
        model.predict(dummy, verbose=False, imgsz=imgsz, device=device)

    t0 = time.perf_counter()
    for fr in frames:
        model.predict(fr, verbose=False, imgsz=imgsz, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return len(frames) / dt, dt / len(frames) * 1000

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = os.path.join(WEIGHTS, "drone.pt")
    if not os.path.exists(src):
        raise SystemExit(f"[错误] 找不到 {src}，请先跑 step3_train_drone.py")
    frames = get_sample_frames()
    print(f"[基准] 用 {len(frames)} 帧测速\n")
    results = []   # (名称, 文件, 大小MB, fps, ms)

    print("=" * 70)
    print("[A] 原始 PyTorch 模型 drone.pt")
    print("=" * 70)
    m = YOLO(src)
    fps, ms = bench(m, frames)
    print(f"  大小 {file_mb(src):.2f} MB   速度 {fps:.1f} fps ({ms:.1f} ms/帧)")
    results.append(("PyTorch fp32", src, file_mb(src), fps, ms))

    try:
        p = m.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True)
        # Ultralytics 每次导出都用同一个名字 drone.onnx，不重命名会互相覆盖
        p_dst = os.path.join(WEIGHTS, "drone_fp32.onnx")
        if os.path.abspath(p) != os.path.abspath(p_dst):
            shutil.move(p, p_dst)
            p = p_dst
        print(f"  导出 -> {p}")
        m2 = YOLO(p)
        fps, ms = bench(m2, frames)
        print(f"  大小 {file_mb(p):.2f} MB   速度 {fps:.1f} fps ({ms:.1f} ms/帧)")
        results.append(("ONNX fp32", p, file_mb(p), fps, ms))
    except Exception as e:
        print(f"  [失败] {e}")

    print("\n" + "=" * 70)
    print("[C] 导出 ONNX (fp16 半精度)")
    print("=" * 70)
    try:
        p16 = m.export(format="onnx", half=True, imgsz=IMGSZ, dynamic=False,
                       simplify=True)
        p16_dst = os.path.join(WEIGHTS, "drone_fp16.onnx")
        if os.path.abspath(p16) != os.path.abspath(p16_dst):
            shutil.move(p16, p16_dst)
            p16 = p16_dst
        print(f"  导出 -> {p16}")
        m3 = YOLO(p16)
        fps, ms = bench(m3, frames)
        print(f"  大小 {file_mb(p16):.2f} MB   速度 {fps:.1f} fps ({ms:.1f} ms/帧)")
        results.append(("ONNX fp16", p16, file_mb(p16), fps, ms))
    except Exception as e:
        print(f"  [失败] {e}")

    print("\n" + "=" * 70)
    print("[D] 导出 ONNX (int8 量化，需要校准数据)")
    print("=" * 70)

    try:
        kw = {}
        if os.path.exists(DATA_YAML):
            kw["data"] = DATA_YAML  # int8 需要数据集做校准
        else:
            print("  [警告] 找不到 data.yaml，int8 校准可能失败")
        p8 = m.export(format="onnx", int8=True, imgsz=IMGSZ, dynamic=False,
                      simplify=True, **kw)
        p8_dst = os.path.join(WEIGHTS, "drone_int8.onnx")
        if os.path.abspath(p8) != os.path.abspath(p8_dst):
            shutil.move(p8, p8_dst)
            p8 = p8_dst
        print(f"  导出 -> {p8}")
        m4 = YOLO(p8)
        fps, ms = bench(m4, frames)
        print(f"  大小 {file_mb(p8):.2f} MB   速度 {fps:.1f} fps ({ms:.1f} ms/帧)")
        results.append(("ONNX int8", p8, file_mb(p8), fps, ms))
    except Exception as e:
        print(f"  [失败] {e}")

    print("\n" + "=" * 70)
    print("[精度验证] 各模型在验证集上的 mAP50")
    print("=" * 70)
    accs = {}
    if os.path.exists(DATA_YAML):
        for name, path, *_ in results:
            try:
                mm = YOLO(path)
                r = mm.val(data=DATA_YAML, imgsz=IMGSZ, verbose=False, plots=False)
                accs[name] = float(r.box.map50)
                print(f"  {name:<14} mAP50 = {accs[name]:.4f}")
            except Exception as e:
                print(f"  {name:<14} 验证失败: {e}")
    else:
        print("  [跳过] 没有 data.yaml")

    names = [r[0] for r in results]
    sizes = [r[2] for r in results]
    fpss = [r[3] for r in results]
    maps = [accs.get(n, 0) for n in names]

    ncol = 3 if accs else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 4.6))
    axes = np.atleast_1d(axes)

    axes[0].bar(names, sizes, color="#4c72b0")
    axes[0].set_title("模型大小 (MB)")
    axes[0].set_ylabel("MB")
    for i, v in enumerate(sizes):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(names, fpss, color="#55a868")
    axes[1].set_title("推理速度 (fps)")
    axes[1].set_ylabel("fps")
    for i, v in enumerate(fpss):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")

    if accs:
        axes[2].bar(names, maps, color="#c44e52")
        axes[2].set_title("精度 mAP50")
        axes[2].set_ylabel("mAP50")
        for i, v in enumerate(maps):
            axes[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        axes[2].grid(alpha=0.3, axis="y")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Task 4 方向1：量化/导出前后对比", fontsize=15)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "quant_compare.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[保存] {out}")

    print(f"\n{'模型':<16}{'大小(MB)':>12}{'fps':>10}{'单帧(ms)':>12}{'mAP50':>10}")
    print("-" * 60)
    for name, path, size, fps, ms in results:
        mp = f"{accs[name]:.4f}" if name in accs else "-"
        print(f"{name:<16}{size:>12.2f}{fps:>10.1f}{ms:>12.1f}{mp:>10}")


if __name__ == "__main__":
    main()
