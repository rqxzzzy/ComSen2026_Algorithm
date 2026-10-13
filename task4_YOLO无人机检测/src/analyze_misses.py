"""分析 drone.pt 在视频上的逐帧检测情况，找出漏检原因。"""
import os
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = r"D:\ComSen2026\task4"
VIDEO = os.path.join(ROOT, "data", "example", "example.mp4")
DRONE = os.path.join(ROOT, "weights", "drone.pt")

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
print(f"视频共 {len(frames)} 帧\n")

model = YOLO(DRONE)

print("用 conf=0.01 跑一遍，记录每帧的最高 drone 置信度...")
rows = []
for i, fr in enumerate(frames):
    r = model.predict(fr, conf=0.01, verbose=False, imgsz=640)[0]
    best = 0.0
    nbox = 0
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            if model.names[int(b.cls.item())] == "drone":
                best = max(best, float(b.conf.item()))
                nbox += 1
    rows.append((i, best, nbox))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(frames)}")

confs = np.array([r[1] for r in rows])

print(f"\n{'=' * 62}")
print("每帧最高 drone 置信度 分布")
print(f"{'=' * 62}")
print(f"  最大值   {confs.max():.3f}")
print(f"  中位数   {np.median(confs):.3f}")
print(f"  最小值   {confs.min():.3f}")
print(f"  均值     {confs.mean():.3f}")
print(f"  完全为 0 的帧（模型压根没看到）: {(confs == 0).sum()} 帧 "
      f"({(confs == 0).mean() * 100:.1f}%)")

print(f"\n{'=' * 62}")
print("不同置信度阈值下的检出率")
print(f"{'=' * 62}")
print(f"{'阈值':>8}{'检出帧数':>12}{'检出率':>12}")
for t in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    hit = (confs >= t).sum()
    print(f"{t:>8.2f}{hit:>12}{hit / len(confs) * 100:>11.1f}%")

print(f"\n{'=' * 62}")
print("conf=0.25 时漏检帧的分布（看是否成片连续，判断原因）")
print(f"{'=' * 62}")
miss = [i for i, c, _ in rows if c < 0.25]
if miss:
    runs = []
    start = prev = miss[0]
    for m in miss[1:]:
        if m == prev + 1:
            prev = m
        else:
            runs.append((start, prev))
            start = prev = m
    runs.append((start, prev))
    print(f"  漏检 {len(miss)} 帧，分成 {len(runs)} 个连续区间：")
    for a, b in runs[:15]:
        print(f"    帧 {a:>4} ~ {b:>4}   共 {b - a + 1:>3} 帧")
    if len(runs) > 15:
        print(f"    ... 还有 {len(runs) - 15} 个区间")
    longest = max(runs, key=lambda r: r[1] - r[0])
    print(f"  最长漏检区间: 帧 {longest[0]} ~ {longest[1]} "
          f"({longest[1] - longest[0] + 1} 帧)")
else:
    print("  没有漏检帧")

mult = [r for r in rows if r[2] > 1]
print(f"\n  一帧内检出多个 drone 的帧数: {len(mult)}")
if mult:
    print(f"    例如: {[(r[0], r[2]) for r in mult[:10]]}")

np.save(os.path.join(ROOT, "outputs", "drone_conf_per_frame.npy"), confs)
print(f"\n[保存] 逐帧置信度 -> outputs/drone_conf_per_frame.npy")
