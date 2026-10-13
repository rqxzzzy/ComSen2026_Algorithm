import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

ROOT = r"D:\ComSen2026\task4"
VIDEO = os.path.join(ROOT, "data", "example", "example.mp4")
WEIGHTS = os.path.join(ROOT, "weights")
OUT_DIR = os.path.join(ROOT, "outputs")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

WARMUP = 10
SAMPLE_N = 30
CLS_COLORS = {}


def color_for(name):
    if name not in CLS_COLORS:
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        CLS_COLORS[name] = tuple(int(v) for v in rng.integers(60, 255, 3))
    return CLS_COLORS[name]


def file_mb(path):
    return os.path.getsize(path) / 1048576 if os.path.exists(path) else 0.0


def warmup_model(model, imgsz=640, n=WARMUP):
    """预热：第一次推理要建 CUDA 上下文，那一次会慢一个数量级，必须剔除。"""
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(n):
        model.predict(dummy, verbose=False, device=0, imgsz=imgsz)


def bench_fps(model, frames, imgsz=640, device=0):
    """纯推理速度基准（不含视频读写和画框开销）。

    前后各 synchronize 一次：GPU 调用是异步的，不同步的话
    计时只量到"把任务塞进队列"的时间，fps 会虚高好几倍。
    """
    warmup_model(model, imgsz=imgsz)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for fr in frames:
        model.predict(fr, verbose=False, device=device, imgsz=imgsz)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return len(frames) / dt if dt > 0 else 0.0, dt / len(frames) * 1000


os.makedirs(OUT_DIR, exist_ok=True)

drone = os.path.join(WEIGHTS, "drone.pt")
if not os.path.exists(drone):
    raise SystemExit(f"没找到 {drone}，请先跑 step3_train_drone.py")

out_video = os.path.join(OUT_DIR, "video_drone_tracked.mp4")
out_table = os.path.join(OUT_DIR, "metrics_table.png")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"打不开视频 {VIDEO}")

# 先采样若干帧，专门做纯推理速度基准
sample = []
for _ in range(SAMPLE_N):
    ok, fr = cap.read()
    if not ok:
        break
    sample.append(fr)

fps_in = cap.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"{'=' * 70}\n无人机检测模型 drone.pt\n{'=' * 70}")
print(f"[视频] {os.path.basename(VIDEO)}  {w}x{h} @ {fps_in:.1f}fps  共 {total} 帧")
print(f"[基准] 采样 {len(sample)} 帧测纯推理速度")

MODEL = YOLO(drone)
fps_b, ms_b = bench_fps(MODEL, sample)

# 回到第 0 帧，逐帧推理并写出画框视频
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

n, hits, all_boxes = 0, 0, []
t0 = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = MODEL.predict(frame, conf=0.25, verbose=False, device=0)[0]

    if r.boxes is not None and len(r.boxes) > 0:
        hits += 1
        for b in r.boxes:
            cls_id = int(b.cls.item())
            name = MODEL.names[cls_id]
            cf = float(b.conf.item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

            c = color_for(name)      # 注意别叫 color：会覆盖上面的函数名，第二次就报 tuple 不可调用
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            label = f"{name} {cf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 6, y1), c, -1)
            cv2.putText(frame, label, (x1 + 3, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            all_boxes.append((n, name, cf, x1, y1, x2, y2))

    cv2.putText(frame, f"frame {n}/{total}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    writer.write(frame)
    n += 1

dt_video = time.perf_counter() - t0
cap.release()
writer.release()

hit_rate = hits / max(n, 1) * 100
print(f"  帧数 {n}  检出帧 {hits}  检出率 {hit_rate:.1f}%")
print(f"  纯推理速度 {fps_b:.1f} fps ({ms_b:.1f} ms/帧)   含读写画框 {n / dt_video:.1f} fps")
print(f"  模型大小 {file_mb(drone):.2f} MB")
print(f"  -> 视频 {out_video}")

if all_boxes:
    from collections import Counter
    c = Counter(b[1] for b in all_boxes)
    print(f"  检测类别分布: {dict(c)}")

rows = [["drone.pt (自训)", "PyTorch .pt", f"{file_mb(drone):.2f}",
         f"{ms_b:.1f}", f"{fps_b:.1f}", f"{hit_rate:.1f}%"]]

fig, ax = plt.subplots(figsize=(11, 1.0 + 0.55 * len(rows)))
ax.axis("off")
cols = ["模型", "格式", "模型大小(MB)", "平均单帧(ms)", "推理速度(fps)", "检出率"]
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.7)
for j in range(len(cols)):
    table[0, j].set_facecolor("#4c72b0")
    table[0, j].set_text_props(color="white", weight="bold")
ax.set_title("Task 4 方向1：推理指标对比", fontsize=14, pad=14)
fig.tight_layout()
fig.savefig(out_table, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[保存] {out_table}")
