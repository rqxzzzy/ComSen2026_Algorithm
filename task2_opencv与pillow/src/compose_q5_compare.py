"""把 q5 四种方法的视频横向拼成一条对比视频。"""
import os

import cv2
import numpy as np

VIDEO_DIR = r"D:\ComSen2026\task2\outputs\videos"
METHODS = ["csrt", "kcf", "mil", "csrt_重检测"]
GAP = 10


def compose(stem):
    paths = [os.path.join(VIDEO_DIR, f"q5_{stem}_{m}.mp4") for m in METHODS]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[跳过] {stem}：缺少 {missing}")
        return

    caps = [cv2.VideoCapture(p) for p in paths]
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30

    out = os.path.join(VIDEO_DIR, f"q5_{stem}_compare.mp4")
    writer = cv2.VideoWriter(
        out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 4 + GAP * 3, h)
    )
    gap = np.full((h, GAP, 3), 255, np.uint8)

    n = 0
    while True:
        frames = []
        alive = False
        for c in caps:
            ok, fr = c.read()
            if ok:
                alive = True
                if fr.shape[0] != h or fr.shape[1] != w:
                    fr = cv2.resize(fr, (w, h))
                frames.append(fr)
            else:
                frames.append(np.zeros((h, w, 3), np.uint8))
        if not alive:
            break
        writer.write(np.hstack([frames[0], gap, frames[1], gap, frames[2], gap, frames[3]]))
        n += 1

    for c in caps:
        c.release()
    writer.release()
    print(f"[保存] {os.path.basename(out)}  {n} 帧  {w * 4 + GAP * 3}x{h}")


for s in ("example3", "example4"):
    compose(s)
