"""追踪失败原因排查（task2 题五）—— 把当时的一组一次性诊断脚本合并成一个，用子命令调用。

背景：example4.mp4 上 CSRT / KCF / MIL 三种追踪器全部失效（KCF 只剩 34 帧有效）。
为定位原因做了几组实验，结论见 REPORT.md 题五：

    素材检查    球长什么样、运动范围多大、什么时候被手挡住
    检测器实验  试了四种手工检测器（霍夫圆、亮度分位数+圆度、top-hat 形态学、V/S 阈值），
                想看能否给 detect-then-track 提供可靠的重检测
    追踪结果    抽帧看追踪框实际跑到哪里去了

结论：**手工检测器全部失败**——木纹背景的边缘与手的轮廓，在面积/圆度上与球处于同一
水平，无法区分。瓶颈不在追踪器而在检测器，正确路线是用深度学习检测器（即 Task 4）。

用法：
    python probe_tracking.py                列出所有子命令
    python probe_tracking.py frames         素材检查：把关键帧拼成网格
    python probe_tracking.py grid           给首帧叠加坐标网格，便于读像素位置
    python probe_tracking.py roi            裁出 example4 首帧人工标定区域 + 颜色统计
    python probe_tracking.py e4frames       抽 example4 关键帧并标出人工初始框
    python probe_tracking.py e4csrt         抽 example4 的 CSRT 结果帧（看追踪框跑哪去了）
    python probe_tracking.py hough          霍夫圆检测参数实验
    python probe_tracking.py bright         亮度分位数阈值 + 圆度
    python probe_tracking.py tophat         top-hat 形态学（对背景明暗变化更鲁棒）
    python probe_tracking.py whiteball      V/S 阈值 + 面积/圆度/长宽比筛选
    python probe_tracking.py detector       诊断 detect_ball 为什么在 example4 上找不到球
"""
import sys

import cv2
import numpy as np

from common import DATA, OUT, VIDEO_OUT, save_grid

E3 = "example3.mp4"
E4 = "example4.mp4"
SCALE_MAX = 960


# ----------------------------------------------------------------- 通用工具

def read_frames(name, idxs, scale_max=SCALE_MAX, driven_by_bg=False):
    """按帧号抽取画面。画幅超过 scale_max 时等比缩小（后续处理快很多）。

    driven_by_bg=True 时同时推进一个 MOG2 背景模型（模拟真实逐帧处理节奏）。
    返回 {帧号: 画面}，读不到的帧号不会出现。
    """
    p = DATA / name
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"[跳过] 打不开 {p}")
        return {}, cap

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, scale_max / max(src_w, src_h))
    w, h = int(src_w * scale), int(src_h * scale)

    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24,
                                            detectShadows=True) if driven_by_bg else None

    out, idx = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if scale < 1.0:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        if idx in idxs:
            out[idx] = f.copy()
        if bg is not None:
            bg.apply(f)          # 让背景模型按正常节奏推进
        idx += 1
    cap.release()
    return out, cap


def frame_size(name):
    """读视频的画幅 / 总帧数 / fps。"""
    cap = cv2.VideoCapture(str(DATA / name))
    if not cap.isOpened():
        return None
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return info


def circles_to_boxes(mask, min_area, max_area, min_circ):
    """二值掩码 -> [(框, 面积, 圆度)]，按 面积×圆度 排序。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        if circ < min_circ:
            continue
        out.append((cv2.boundingRect(c), area, circ))
    out.sort(key=lambda t: t[1] * t[2], reverse=True)
    return out


def draw_candidates(frame, cands, limit=4, thickness=3, show="circ"):
    """在画面上框出候选（第一个绿、其余橙），并标上圆度。"""
    vis = frame.copy()
    for i, (box, area, circ) in enumerate(cands[:limit]):
        x, y, bw, bh = box
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, thickness)
        label = f"{circ:.2f}" if show == "circ" else f"{area:.0f}"
        cv2.putText(vis, label, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------- 素材检查

def cmd_frames(names=(E3, E4)):
    """把每个视频的 4 个等分点关键帧拼成网格图。"""
    for name in names:
        p = DATA / name
        if not p.exists():
            print(f"[跳过] {name} 不存在")
            continue
        info = frame_size(name)
        if info is None:
            print(f"[跳过] {name} 无法打开")
            continue
        w, h, total, fps = info
        print(f"[读取] {name}  {w}x{h} @ {fps:.1f}fps  {total} 帧")

        cap = cv2.VideoCapture(str(p))
        frames, titles = [], []
        for frac in (0.0, 0.25, 0.5, 0.75):
            idx = min(int(total * frac), max(0, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if ok:
                frames.append(f)
                titles.append(f"{name} 第 {idx} 帧")
        cap.release()

        if frames:
            save_grid(frames, titles, f"probe_{p.stem}.png", cols=2,
                      figsize=(15, 4.3 * len(frames) / 2),
                      suptitle=f"素材检查：{name}")


def cmd_grid(names=(E4, E3), frame_idx=0, step=100):
    """给首帧叠加坐标网格并把刻度写在线上，便于直接读出目标的像素坐标。"""
    for name in names:
        p = DATA / name
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"[跳过] {name} 打不开")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, f = cap.read()
        cap.release()
        if not ok:
            print(f"[跳过] {name} 读取失败")
            continue

        h, w = f.shape[:2]
        canvas = f.copy()
        for x in range(0, w, step):
            cv2.line(canvas, (x, 0), (x, h), (0, 255, 0), 1)
            cv2.putText(canvas, str(x), (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(canvas, (0, y), (w, y), (0, 255, 0), 1)
            cv2.putText(canvas, str(y), (4, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

        out = OUT / f"probe_{p.stem}_grid.jpg"
        cv2.imwrite(str(out), canvas)
        print(f"[保存] {out}   ({w}x{h}, 网格 {step}px)")


def cmd_roi(video=E4):
    """裁出人工标定区域，并打印它的平均 HSV/BGR——用来判断白球到底长什么样。"""
    from q5_tracker import MANUAL_ROI
    box = MANUAL_ROI.get(video)
    if box is None:
        print(f"[跳过] {video} 没有人工标定框")
        return

    cap = cv2.VideoCapture(str(DATA / video))
    ok, f = cap.read()
    cap.release()
    if not ok:
        print("读取失败")
        return

    h, w = f.shape[:2]
    print(f"画幅: {w}x{h}")

    x, y, bw, bh = box
    roi = f[y:y + bh, x:x + bw]
    cv2.imwrite(str(OUT / "dbg_e4_roi.jpg"), roi)
    print(f"ROI 裁切: ({x},{y}) 尺寸 {bw}x{bh} -> dbg_e4_roi.jpg")

    cv2.imwrite(str(OUT / "dbg_e4_bottomright.jpg"), f[600:1080, 1200:1920])
    print("右下区域裁切: (1200,600)-(1920,1080) -> dbg_e4_bottomright.jpg")

    canvas = f.copy()
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 6)
    cv2.putText(canvas, "MANUAL_ROI", (max(0, x - 420), max(40, y - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "dbg_e4_box.jpg"), canvas)
    print("整帧标注 -> dbg_e4_box.jpg")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    bgr = roi.reshape(-1, 3).mean(axis=0)
    print(f"ROI 平均 HSV = {hsv.round(1)}   (白球应为高 V 低 S)")
    print(f"ROI 平均 BGR = {bgr.round(1)}   (白球三通道应接近且值高)")


def cmd_e4frames(video=E4, fracs=(0.0, 0.15, 0.3, 0.5, 0.7, 0.9)):
    """抽关键帧并画出人工初始框，看球的运动范围有多大。"""
    from q5_tracker import MANUAL_ROI
    info = frame_size(video)
    if info is None:
        print(f"[跳过] {video} 打不开")
        return
    w, h, total, fps = info
    print(f"{video}  {w}x{h} @ {fps:.1f}fps  {total} 帧")

    roi = MANUAL_ROI.get(video)
    print(f"MANUAL_ROI = {roi}")

    cap = cv2.VideoCapture(str(DATA / video))
    frames, titles = [], []
    for frac in fracs:
        idx = min(int(total * frac), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        if roi:
            x, y, bw, bh = roi
            cv2.rectangle(f, (x, y), (x + bw, y + bh), (0, 255, 0), 6)
            cv2.putText(f, "MANUAL_ROI", (max(0, x - 460), max(50, y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5, cv2.LINE_AA)
        frames.append(f)
        titles.append(f"第 {idx} 帧（t={frac:.0%}）")
    cap.release()

    save_grid(frames, titles, "probe_e4_frames.png", cols=2, figsize=(16, 13),
              suptitle=f"{video}：白球运动范围与人工初始框（绿框）")


def cmd_e4csrt(video=E4, idxs=(0, 50, 150, 300, 450, -1)):
    """抽追踪结果视频的若干帧——看追踪框到底跑哪去了。

    读取的是 q5_tracker 输出的画框视频（outputs/videos/ 下），
    因此需要先跑过题五才能用这个子命令。
    """
    src = VIDEO_OUT / f"q5_{video.split('.')[0]}_csrt.mp4"
    if not src.exists():
        print(f"[跳过] 找不到追踪结果视频 {src}\n"
              f"        先跑 `python q5_tracker.py` 生成，或该视频只随邮件附件提交、不在仓库内。")
        return

    cap = cv2.VideoCapture(str(src))
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    w, h, total = info
    print(f"{src}\n  {w}x{h}  共 {total} 帧")

    frames, titles = [], []
    for i in idxs:
        i = max(0, total - 11) if i < 0 else i
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
            titles.append(f"第 {i} 帧")
    cap.release()

    if frames:
        save_grid(frames, titles, "probe_e4_csrt.png", cols=2, figsize=(16, 13),
                  suptitle=f"{video} — CSRT 追踪结果（绿框是 tracker 的当前输出）")


# ------------------------------------------------------------- 检测器实验
# 下面四个都是「能不能用手工特征可靠地重检测白球」的尝试，结论均为失败。

def cmd_hough(cases=(E3, E4), frame_idxs=(0, 60)):
    """霍夫圆检测：换几组参数，看能否用统一策略定位两种素材的小球。"""
    for name in cases:
        cap = cv2.VideoCapture(str(DATA / name))
        for frame_idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[{name}] 第 {frame_idx} 帧读取失败")
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            short = min(h, w)

            print(f"\n===== {name}  {w}x{h}  第 {frame_idx} 帧 =====")
            for dp, p1, p2 in ((1.2, 100, 30), (1.2, 100, 50), (1.5, 120, 40), (1.2, 80, 25)):
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=int(short * 0.15),
                    param1=p1, param2=p2,
                    minRadius=int(short * 0.02), maxRadius=int(short * 0.30),
                )
                tag = f"dp={dp} p1={p1} p2={p2}"
                if circles is None:
                    print(f"  {tag:<28} 无结果")
                    continue
                found = [tuple(int(v) for v in c) for c in circles[0][:4]]
                print(f"  {tag:<28} {len(circles[0])} 个: "
                      + ", ".join(f"(x={x},y={y},r={r})" for x, y, r in found))
        cap.release()


def cmd_bright(video=E4, frames=(0, 60, 120, 200, 300, 400, 500, 570),
               pct=97.0, min_area=400, max_area=40000, min_circ=0.55):
    """亮度分位数阈值 + 饱和度上限 + 圆度筛选。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    got, _ = read_frames(video, set(frames))
    if not got:
        return

    panels, titles = [], []
    for fidx in frames:
        if fidx not in got:
            continue
        frame = got[fidx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, sat, _ = cv2.split(hsv)

        thr = np.percentile(gray, pct)
        mask = ((gray >= thr) & (sat < 90)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cands = circles_to_boxes(mask, min_area, max_area, min_circ)
        print(f"第 {fidx:>3} 帧  阈值={thr:>5.0f}  候选 {len(cands)} 个: "
              + (" | ".join(f"圆度{c:.2f} 框{box}" for box, a, c in cands[:3]) or "无"))

        panels.append(draw_candidates(frame, cands))
        titles.append(f"第 {fidx} 帧（阈值={thr:.0f}，{len(cands)} 个候选）")

    save_grid(panels, titles, "probe_bright.png", cols=2,
              figsize=(16, 4.2 * ((len(panels) + 1) // 2)),
              suptitle=f"{video}：亮度阈值+圆度 的候选框（绿=最优候选）")


def cmd_tophat(video=E4, frames=(0, 60, 120, 200, 300, 400, 500, 570),
               ksize=51, thresh=18, min_area=200, max_area=40000, min_circ=0.45):
    """top-hat 形态学：专门找「纹理背景上的局部高亮小目标」，对背景明暗变化更鲁棒。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    got, _ = read_frames(video, set(frames))
    if not got:
        return

    panels, titles = [], []
    for fidx in frames:
        if fidx not in got:
            continue
        frame = got[fidx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        _, mask = cv2.threshold(top, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk, iterations=2)

        cands = circles_to_boxes(mask, min_area, max_area, min_circ)
        print(f"第 {fidx:>3} 帧  候选 {len(cands)} 个: "
              + (" | ".join(f"圆度{c:.2f} 面积{a:.0f} 框{box}"
                            for box, a, c in cands[:3]) or "无"))

        panels.append(draw_candidates(frame, cands))
        titles.append(f"第 {fidx} 帧（{len(cands)} 个候选）")
        # 把 top-hat 响应归一化后并排展示
        top_vis = cv2.normalize(top, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        panels.append(cv2.cvtColor(top_vis, cv2.COLOR_GRAY2BGR))
        titles.append(f"第 {fidx} 帧 top-hat 响应")

    save_grid(panels, titles, "probe_tophat.png", cols=2,
              figsize=(16, 4.0 * ((len(panels) + 1) // 2)),
              suptitle=f"{video}：top-hat 检测结果（绿=最优候选）")


def cmd_whiteball(video=E4, frames=(0, 60, 145, 290, 435, 580),
                  grids=(("v200_s60", 200, 60), ("v180_s70", 180, 70),
                         ("v160_s80", 160, 80)),
                  min_area=1500, min_circ=0.45, min_aspect=0.6, max_aspect=1.7):
    """V/S 阈值 + 面积/圆度/长宽比筛选。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cap = cv2.VideoCapture(str(DATA / video))

    for tag, vt, st in grids:
        print(f"\n########## 阈值 V>{vt}, S<{st} ##########")
        for idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _, s, v = cv2.split(hsv)
            mask = ((v > vt) & (s < st)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cands = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                peri = cv2.arcLength(c, True)
                circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / bh if bh else 0.0
                if circ >= min_circ and min_aspect <= aspect <= max_aspect:
                    cands.append((area, circ, (x, y, bw, bh)))
            cands.sort(key=lambda t: t[0], reverse=True)

            fh, fw = frame.shape[:2]
            desc = " | ".join(f"面积{a:.0f} 圆度{c:.2f} 框{box}"
                              for a, c, box in cands[:2]) or "无候选"
            print(f"  帧{idx:>4} (画幅 {fw}x{fh}): {desc}")
    cap.release()


def cmd_detector(video=E4, frames=(0, 30, 60, 120, 200, 300, 400, 500, 575)):
    """诊断 q5_tracker.detect_ball：逐条列出掩码里的连通域，并说明为什么被筛掉。

    这是定位「检测器不可靠」最直接的一块证据——被判定为候选的往往是木纹边缘和手的
    轮廓，它们的面积/圆度/颜色与球处在同一水平。
    """
    from q5_tracker import (MORPH_KERNEL, DET_MIN_AREA, DET_MAX_AREA,
                            DET_MIN_CIRCULARITY, DET_MIN_COLOR_SIM, MANUAL_ROI)

    src_w, src_h = frame_size(video)[:2]
    scale = min(1.0, SCALE_MAX / max(src_w, src_h))
    w, h = int(src_w * scale), int(src_h * scale)
    print(f"{DATA / video}  {src_w}x{src_h} -> {w}x{h}")
    print(f"人工标定初始框(原画幅) = {MANUAL_ROI.get(video)}")
    print(f"筛选条件: 面积[{DET_MIN_AREA}, {DET_MAX_AREA}]  "
          f"圆度>={DET_MIN_CIRCULARITY}  颜色相似度>={DET_MIN_COLOR_SIM}\n")

    got, _ = read_frames(video, set(frames), driven_by_bg=True)
    if not got:
        return
    # 掩码必须和画面同分辨率：重建背景模型时也要做同样的缩放
    fh, fw = h, w

    # 参照颜色：首帧人工框内的 HSV 均值（与 detect_ball 的 ref_color 同口径：
    # MANUAL_ROI 存的是原始画幅坐标，用之前要按缩放比例换算）
    ref_s = ref_v = None
    box = MANUAL_ROI.get(video)
    if 0 in got and box:
        rx, ry, rw, rh = [int(v * scale) for v in box]
        roi = cv2.cvtColor(got[0][ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2HSV)
        if roi.size:
            ref_s, ref_v = float(roi[:, :, 1].mean()), float(roi[:, :, 2].mean())
    print(f"参照颜色(首帧人工框): S={ref_s:.1f} V={ref_v:.1f}\n")

    for fidx in frames:
        if fidx not in got:
            continue
        frame = got[fidx]

        # 重新建一个背景模型跑到该帧，保证掩码状态与逐帧处理时一致（同样要缩放）
        bg2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24,
                                                 detectShadows=True)
        cap2 = cv2.VideoCapture(str(DATA / video))
        mask = None
        for _ in range(fidx + 1):
            ok, f = cap2.read()
            if not ok:
                break
            if f.shape[:2] != (fh, fw):
                f = cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA)
            mask = bg2.apply(f)
        cap2.release()
        if mask is None:
            continue

        mask = mask.copy()
        mask[mask == 127] = 0            # 阴影像素丢掉
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        print(f"===== 第 {fidx} 帧：掩码里 {len(contours)} 个连通域 =====")
        rows = []
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
            x, y, bw, bh = cv2.boundingRect(c)
            rs = sat[y:y + bh, x:x + bw]
            rv = val[y:y + bh, x:x + bw]
            if rs.size and ref_s is not None:
                ds = abs(float(rs.mean()) - ref_s) / 255.0
                dv = abs(float(rv.mean()) - ref_v) / 255.0
                color_sim = max(0.0, 1.0 - (ds * 0.5 + dv * 0.5) * 2.0)
            else:
                color_sim = 0.0
            ok_area = DET_MIN_AREA <= area <= DET_MAX_AREA
            ok_circ = circ >= DET_MIN_CIRCULARITY
            ok_color = color_sim >= DET_MIN_COLOR_SIM
            rows.append((ok_area and ok_circ and ok_color,
                         area, circ, color_sim, x, y, bw, bh,
                         (ok_area, ok_circ, ok_color)))

        rows.sort(key=lambda r: (not r[0], -r[1]))
        for passed, area, circ, color_sim, x, y, bw, bh, cond in rows[:8]:
            mark = "✅通过" if passed else "  ✗   "
            why = ""
            if not passed:
                bad = []
                if not cond[0]:
                    bad.append("面积越界")
                if not cond[1]:
                    bad.append("不够圆")
                if not cond[2]:
                    bad.append("颜色不像")
                why = "  <- " + "/".join(bad)
            print(f"  {mark} 面积={area:>8.0f} 圆度={circ:.2f} 色似={color_sim:.2f}"
                  f"  框=({x},{y},{bw},{bh}){why}")
        print()


# ----------------------------------------------------------------- 入口

COMMANDS = {
    "frames":    (cmd_frames,    "素材检查：关键帧网格"),
    "grid":      (cmd_grid,      "首帧坐标网格"),
    "roi":       (cmd_roi,       "人工标定区域 + 颜色统计"),
    "e4frames":  (cmd_e4frames,  "example4 关键帧 + 人工初始框"),
    "e4csrt":    (cmd_e4csrt,    "example4 CSRT 结果抽帧"),
    "hough":     (cmd_hough,     "霍夫圆检测参数实验"),
    "bright":    (cmd_bright,    "亮度分位数阈值 + 圆度"),
    "tophat":    (cmd_tophat,    "top-hat 形态学"),
    "whiteball": (cmd_whiteball, "V/S 阈值 + 圆度/长宽比"),
    "detector":  (cmd_detector,  "detect_ball 为什么找不到球"),
}


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "list"):
        print(__doc__)
        print("可用子命令：")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<10} {desc}")
        return 0

    name = argv[0]
    if name not in COMMANDS:
        print(f"未知子命令: {name}\n")
        for n, (_, desc) in COMMANDS.items():
            print(f"  {n:<10} {desc}")
        return 1

    COMMANDS[name][0](*argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
