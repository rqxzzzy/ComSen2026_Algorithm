"""题五：用 OpenCV 追踪器追踪视频中的小球，对比多种提升追踪准确度的方法。"""
import csv

import cv2
import numpy as np

from common import DATA, OUT, VIDEO_OUT, resolve_video
import matplotlib.pyplot as plt

VIDEO_LIST = ["example3.mp4", "example4.mp4"]
TRACKER_LIST = ["CSRT", "KCF", "MIL", "CSRT+重检测"]

# 自动定位失败时的兜底框（在原始分辨率下测量）
MANUAL_ROI = {
    "example4.mp4": (1445, 785, 290, 290),
    "example3.mp4": (299, 11, 113, 113),
}

SCALE_MAX = 960          # 处理前把过大画幅缩到最长边不超过此值
KERNEL = np.ones((5, 5), np.uint8)
MIN_AREA, MAX_AREA = 300, 40000     # 仅供调试可视化 stages() 使用
MIN_CIRC = 0.55                     # 圆度下限，4πA/P²，正圆为 1
MIN_COLOR_SIM = 0.45                # 候选颜色与首帧目标颜色的相似度下限
REDETECT_EVERY = 10                 # 每多少帧重检测一次
IOU_REINIT = 0.30                   # 检测框与追踪框 IoU 低于此值就重新初始化

COLORS = {"CSRT": "#2ca02c", "KCF": "#1f77b4", "MIL": "#ff7f0e", "CSRT+重检测": "#d62728"}


def roi_hist(frame, box):
    if box is None:
        return None
    x, y, w, h = [int(v) for v in box]
    x, y = max(0, x), max(0, y)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def ref_sv(frame, box):
    x, y, w, h = [int(v) for v in box]
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()), float(hsv[:, :, 2].mean())


def iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_by_hough(frame):
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    short = min(gray.shape)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(short * 0.15),
        param1=100, param2=30,
        minRadius=int(short * 0.02), maxRadius=int(short * 0.30),
    )
    if circles is None:
        return None
    cx, cy, r = max(circles[0], key=lambda c: c[2])
    cx, cy, r = int(cx), int(cy), int(r)
    return (cx - r, cy - r, 2 * r, 2 * r)


def detect_by_color(frame, min_area=120):
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                       (0, 90, 90), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    return cv2.boundingRect(cnt)


def resolve_init_box(frame, video_name):
    hough = detect_by_hough(frame)
    color = detect_by_color(frame)
    if hough is not None and color is not None and iou(hough, color) >= 0.5:
        box = tuple(int((hough[i] + color[i]) / 2) for i in range(4))
        return box, "霍夫圆+HSV 互相印证"
    if video_name in MANUAL_ROI:
        return MANUAL_ROI[video_name], "人工标定"
    return None, "全部失败"


def detect_ball(frame, bg_sub, ref_color, prev_box, ref_box):
    area0 = float(ref_box[2] * ref_box[3])
    min_area, max_area = area0 * 0.15, area0 * 3.0

    mask = bg_sub.apply(frame)
    mask[mask == 127] = 0                      # 去掉 MOG2 标出的阴影
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    fh, fw = frame.shape[:2]
    ref_s, ref_v = ref_color

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        if circ < MIN_CIRC:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi_s, roi_v = s_ch[y:y + h, x:x + w], v_ch[y:y + h, x:x + w]
        ds = abs(float(roi_s.mean()) - ref_s) / 255.0
        dv = abs(float(roi_v.mean()) - ref_v) / 255.0
        color_sim = max(0.0, 1.0 - (ds * 0.5 + dv * 0.5) * 2.0)
        if color_sim < MIN_COLOR_SIM:
            continue
        score = circ * 0.4 + color_sim * 0.6
        if prev_box is not None:
            pcx, pcy = prev_box[0] + prev_box[2] / 2, prev_box[1] + prev_box[3] / 2
            cx, cy = x + w / 2, y + h / 2
            score -= np.hypot(cx - pcx, cy - pcy) / np.hypot(fw, fh) * 0.5
        if score > best_score:
            best_score, best = score, (x, y, w, h)
    return best


def stages(frame, bg_sub, ref_color, prev_box, verbose=True):
    m0 = bg_sub.apply(frame)
    m1 = m0.copy()
    m1[m1 == 127] = 0
    m2 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, KERNEL)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    fh, fw = frame.shape[:2]
    ref_s, ref_v = ref_color

    vis_cand = frame.copy()
    cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept, dropped = [], {"面积": 0, "圆度": 0, "颜色": 0}

    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            dropped["面积"] += 1
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        if circ < MIN_CIRC:
            dropped["圆度"] += 1
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi_s, roi_v = s_ch[y:y + h, x:x + w], v_ch[y:y + h, x:x + w]
        ds = abs(float(roi_s.mean()) - ref_s) / 255.0
        dv = abs(float(roi_v.mean()) - ref_v) / 255.0
        color_sim = max(0.0, 1.0 - (ds * 0.5 + dv * 0.5) * 2.0)
        if color_sim < MIN_COLOR_SIM:
            dropped["颜色"] += 1
            continue
        kept.append((c, area, circ, color_sim, (x, y, w, h)))

    for c, *_ in kept:
        cv2.rectangle(vis_cand, cv2.boundingRect(c), (0, 255, 255), 3)

    best, best_score = None, -1.0
    for c, area, circ, color_sim, box in kept:
        score = circ * 0.4 + color_sim * 0.6
        if prev_box is not None:
            pcx, pcy = prev_box[0] + prev_box[2] / 2, prev_box[1] + prev_box[3] / 2
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            score -= np.hypot(cx - pcx, cy - pcy) / np.hypot(fw, fh) * 0.5
        if score > best_score:
            best_score, best = score, box

    vis_final = frame.copy()
    if best is not None:
        x, y, w, h = best
        cv2.rectangle(vis_final, (x, y), (x + w, y + h), (0, 255, 0), 4)

    if verbose:
        print(f"    轮廓总数 {len(cnts)}  →  面积过滤掉 {dropped['面积']}"
              f"  →  圆度过滤掉 {dropped['圆度']}  →  颜色过滤掉 {dropped['颜色']}"
              f"  →  通过 {len(kept)}  →  选中 {best}")

    def to_bgr(m):
        return m if m.ndim == 3 else cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    row = [frame, to_bgr(m0), to_bgr(m1), to_bgr(m2), vis_cand, vis_final]
    titles = ["1.原帧", "2.MOG2原始mask", "3.去阴影(去掉127)", "4.形态学后",
              "5.三重筛选通过的候选(黄)", "6.最终选中(绿)"]
    th = 300
    out = []
    for img, t in zip(row, titles):
        im = cv2.resize(img, (int(img.shape[1] * th / img.shape[0]), th))
        cv2.putText(im, t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        out.append(im)
    return out


def make_tracker(name):
    base = name.split("+")[0]
    factories = {
        "CSRT": ("TrackerCSRT_create", "legacy_TrackerCSRT_create"),
        "KCF": ("TrackerKCF_create", "legacy_TrackerKCF_create"),
        "MIL": ("TrackerMIL_create", "legacy_TrackerMIL_create"),
    }
    for attr in factories[base]:
        fn = getattr(cv2, attr, None)
        if fn is not None:
            return fn()
    raise RuntimeError(f"当前 OpenCV({cv2.__version__}) 不支持 tracker: {base}；"
                       f"请安装 opencv-contrib-python")


def run(video_name, tracker_name):
    src = resolve_video(video_name)
    if src is None:
        print(f"  缺少素材 {video_name}，跳过")
        return None

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  打不开 {src}")
        return None

    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return None

    init_bbox, method = resolve_init_box(frame0, video_name)
    print(f"\n====== {video_name} | {tracker_name} ======")
    print(f"  初始框来源：{method}  {init_bbox}")

    # 大画幅先缩放，追踪器在处理大尺寸时会明显变慢
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, SCALE_MAX / max(src_w, src_h))
    size = (int(src_w * scale), int(src_h * scale))
    need_resize = scale < 1.0
    if need_resize:
        frame0 = cv2.resize(frame0, size)
    init_bbox = tuple(int(v * scale) for v in init_bbox)

    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    bg.apply(frame0)
    ref_sv_val = ref_sv(frame0, init_bbox)
    ref_hist = roi_hist(frame0, init_bbox)

    tracker = make_tracker(tracker_name)
    tracker.init(frame0, init_bbox)
    use_redetect = ("+" in tracker_name)     # 只有名字带 + 的方法才启用纠偏

    out_file = f"q5_{video_name[:-4]}_{tracker_name.replace('+', '_').lower()}.mp4"
    writer = cv2.VideoWriter(str(VIDEO_OUT / out_file),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             cap.get(cv2.CAP_PROP_FPS), size)

    rec = {"frames": [], "sims": [], "lost": 0, "boxes": []}
    n = 0
    det_calls = det_hits = reinits = 0
    prev = tuple(float(v) for v in init_bbox)
    pending = frame0          # 首帧也要过一遍 update：让追踪器在起点目标上确认一次模型

    while True:
        if pending is not None:
            frame, pending = pending, None
        else:
            ret, frame = cap.read()
            if not ret:
                break
        if need_resize:
            frame = cv2.resize(frame, size)
        n += 1

        ok, box = tracker.update(frame)
        if ok:
            box = tuple(float(v) for v in box)
            h = roi_hist(frame, box)
            sim = cv2.compareHist(ref_hist, h, cv2.HISTCMP_CORREL) if h is not None else np.nan
        else:
            rec["lost"] += 1
            box = prev                       # 丢失时沿用上一帧框，别让指标断档
            sim = np.nan

        if use_redetect and n % REDETECT_EVERY == 0:
            det_calls += 1
            det = detect_ball(frame, bg, ref_sv_val, prev, init_bbox)
            if det is not None:
                det_hits += 1
                if iou(det, box) < IOU_REINIT:      # 只在严重不一致时才纠偏
                    reinits += 1
                    tracker = make_tracker(tracker_name)
                    tracker.init(frame, det)
                    box = tuple(float(v) for v in det)
        else:
            bg.apply(frame)               # 非检测帧也要喂背景模型，保持状态一致

        rec["frames"].append(n)
        rec["sims"].append(sim)
        rec["boxes"].append(box if ok else None)

        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, tracker_name, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        writer.write(frame)
        prev = box

    cap.release()
    writer.release()
    if use_redetect:
        print(f"  重检测 {det_calls} 次，检测到目标 {det_hits} 次，触发纠偏 {reinits} 次")
    print(f"  丢失 {rec['lost']} 帧 / 共 {n} 帧   →  {VIDEO_OUT / out_file}")
    return rec


def save_curve(runs):
    fig, axes = plt.subplots(1, len(VIDEO_LIST), figsize=(15, 5), squeeze=False)
    for ax, vid in zip(axes[0], VIDEO_LIST):
        for trk in TRACKER_LIST:
            r = runs.get((vid, trk))
            if r is None:
                continue
            ax.plot(r["frames"], r["sims"], label=trk, color=COLORS[trk],
                    linewidth=1.3, alpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(vid, fontsize=12)
        ax.set_xlabel("帧序号")
        ax.set_ylabel("外观一致性（与首帧的直方图相关性）")
        ax.set_ylim(-0.3, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("题五：四种追踪器/方法的外观一致性随时间变化", fontsize=14)
    fig.tight_layout()
    p = OUT / "q5_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[保存] {p}")


def main():
    runs = {}
    for vid in VIDEO_LIST:
        for trk in TRACKER_LIST:
            r = run(vid, trk)
            if r is not None:
                runs[(vid, trk)] = r

    if not runs:
        print("没有任何结果，检查素材是否在 task2/data/ 下")
        return

    p = OUT / "q5_summary.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["视频", "追踪器", "平均一致性", "有效帧", "总帧", "丢失帧", "丢失率"])
        for (vid, trk), r in runs.items():
            s = np.array(r["sims"], dtype=float)
            valid = int(np.sum(~np.isnan(s)))
            avg = float(np.nanmean(s)) if valid else float("nan")
            total = len(s)
            wr.writerow([vid, trk, f"{avg:.4f}", valid, total, r["lost"],
                         f"{r['lost'] / total * 100:.1f}%"])
    print(f"[保存] {p}")

    p = OUT / "q5_perframe.csv"
    max_len = max(len(r["frames"]) for r in runs.values())
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["frame"] + [f"{v[:-4]}_{t}" for (v, t) in runs.keys()])
        for i in range(max_len):
            row = [i + 1]
            for r in runs.values():
                s = r["sims"]
                row.append(f"{s[i]:.4f}" if i < len(s) and not np.isnan(s[i]) else "")
            wr.writerow(row)
    print(f"[保存] {p}")

    save_curve(runs)

    for vid in VIDEO_LIST:
        stem = vid[:-4]
        methods = [t for t in TRACKER_LIST if (vid, t) in runs]
        if not methods:
            continue

        p = OUT / f"q5_{stem}_stats.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["frame"] + methods)
            nlen = max(len(runs[(vid, m)]["sims"]) for m in methods)
            for i in range(nlen):
                row = [i + 1]
                for m in methods:
                    s = runs[(vid, m)]["sims"]
                    row.append(f"{s[i]:.4f}" if i < len(s) and not np.isnan(s[i]) else "")
                wr.writerow(row)
        print(f"[保存] {p}")

        # 曲线 + 柱状
        fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
        for m in methods:
            axes[0].plot(runs[(vid, m)]["sims"], label=m, color=COLORS[m], linewidth=1.2)
        axes[0].axhline(0.6, color="red", linestyle="--", linewidth=1, label="阈值 0.6")
        axes[0].set_title("逐帧外观一致性（框内颜色分布 vs 首帧模板）")
        axes[0].set_xlabel("帧序号")
        axes[0].set_ylabel("直方图相关性")
        axes[0].set_ylim(-0.3, 1.05)
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        means = [float(np.nanmean(np.array(runs[(vid, m)]["sims"], dtype=float)))
                 for m in methods]
        xpos = np.arange(len(methods))
        axes[1].bar(xpos, means, color="#4c9be8")
        axes[1].set_xticks(xpos)
        axes[1].set_xticklabels(methods, rotation=15)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("平均外观一致性（越高越准）")
        axes[1].grid(alpha=0.3, axis="y")
        for i, v in enumerate(means):
            axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

        fig.suptitle(f"题五：追踪准确度对比（{vid}）", fontsize=15)
        fig.tight_layout()
        p = OUT / f"q5_{stem}_metrics.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[保存] {p}")

        # 轨迹
        fig, ax = plt.subplots(figsize=(10, 7))
        for m in methods:
            pts = [((b[0] + b[2] / 2), (b[1] + b[3] / 2)) if b else (np.nan, np.nan)
                   for b in runs[(vid, m)]["boxes"]]
            cx, cy = zip(*pts)
            ax.plot(cx, cy, linewidth=1.5, label=m, color=COLORS[m], alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlim(0, size_of(vid)[0])
        ax.set_ylim(size_of(vid)[1], 0)
        ax.set_xlabel("x 像素")
        ax.set_ylabel("y 像素")
        ax.set_title(f"题五：追踪轨迹对比（{vid}）")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = OUT / f"q5_{stem}_trajectory.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[保存] {p}")

    print("\n===== 全部实验汇总 =====")
    for (vid, trk), r in runs.items():
        s = np.array(r["sims"], dtype=float)
        valid = int(np.sum(~np.isnan(s)))
        avg = float(np.nanmean(s)) if valid else float("nan")
        print(f"  {vid:14s} {trk:10s} 平均一致性 {avg:.4f}  "
              f"有效帧 {valid:>3}/{len(s)}  丢失 {r['lost']}")


_SIZE_CACHE = {}


def size_of(video_name):
    if video_name not in _SIZE_CACHE:
        cap = cv2.VideoCapture(str(resolve_video(video_name)))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, SCALE_MAX / max(w, h))
        _SIZE_CACHE[video_name] = (int(w * scale), int(h * scale))
        cap.release()
    return _SIZE_CACHE[video_name]


if __name__ == "__main__":
    main()
