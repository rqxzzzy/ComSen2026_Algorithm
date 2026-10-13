import csv
import sys

import cv2
import numpy as np

from common import DATA, OUT, VIDEO_OUT, save_grid

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
DIFF_THRESHOLD = 40      # 帧差法二值化阈值（25 太敏感，会把压缩噪声也算成运动）
MIN_BLOB_AREA = 300      # 前景连通域面积下限，滤掉噪声


def make_bg_subtractor():
    """MOG2 背景建模器：history 影响背景更新速度，varThreshold 控制判定灵敏度。"""
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )


def clean_mask(mask, remove_shadow=True):
    """形态学去噪 + 去掉 MOG2 的阴影像素（灰度 127）。"""
    if remove_shadow:
        mask[mask == 127] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return mask


def largest_blobs(mask, min_area=MIN_BLOB_AREA, top_n=3):
    """返回面积最大的若干个前景连通域外接框。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            boxes.append((cv2.boundingRect(c), area))
    boxes.sort(key=lambda t: t[1], reverse=True)
    return boxes[:top_n]


def blob_metrics(boxes):
    """最大团块的面积，以及它的填充率（轮廓面积 / 外接矩形面积）。

    填充率是区分两种前景掩码的关键指标：帧差法只能标出物体「发生变化的边缘」，
    运动物体内部前后帧没变化所以不会变白，团块是空心的 —— 填充率低（实测约 0.49）；
    MOG2 维护完整背景模型，整个物体都被标为前景 —— 填充率高（实测约 0.64）。
    正圆是 0.785，实心矩形是 1.0。
    """
    if not boxes:
        return 0.0, 0.0
    (x, y, w, h), area = boxes[0]
    fill = float(area / (w * h)) if w * h else 0.0
    return float(area), fill


def annotate(frame, boxes, label, color=(0, 255, 0)):
    """把检测框画到帧上，并在左上角写方法名。"""
    out = frame.copy()
    for (x, y, w, h), area in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{int(area)}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(video_name="example2.mp4", sample_name="sample_motion.mp4"):
    src = DATA / video_name
    if not src.exists():
        src = DATA / sample_name
        if not src.exists():
            print(f"[错误] 找不到 {video_name}，也没有替代素材 {sample_name}")
            return
        print(f"[提示] 未找到 {video_name}，改用 {sample_name} 演示")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"无法打开视频 {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[读取] {src.name}  {w}x{h} @ {fps:.1f}fps  共 {total} 帧")

    stem = src.stem
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_diff = cv2.VideoWriter(str(VIDEO_OUT / f"q4_{stem}_framediff.mp4"),
                              fourcc, fps, (w, h))
    vw_mog2 = cv2.VideoWriter(str(VIDEO_OUT / f"q4_{stem}_mog2.mp4"),
                              fourcc, fps, (w, h))
    vw_cmp = cv2.VideoWriter(str(VIDEO_OUT / f"q4_{stem}_compare.mp4"),
                             fourcc, fps, (w * 2 + 10, h))

    bg_sub = make_bg_subtractor()
    prev_gray = None
    stats = []
    key_frames = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 方法 A：帧差法（不使用背景减除）
        if prev_gray is None:
            mask_diff = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(prev_gray, gray)
            _, mask_diff = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            mask_diff = clean_mask(mask_diff, remove_shadow=False)
        prev_gray = gray

        # 方法 B：MOG2 背景减除
        mask_mog2 = bg_sub.apply(frame)
        mask_mog2 = clean_mask(mask_mog2, remove_shadow=True)

        boxes_diff = largest_blobs(mask_diff)
        boxes_mog2 = largest_blobs(mask_mog2)

        # 帧差法只保留左上角的方法标签，不画检测框
        ann_diff = annotate(frame, [], f"A 帧差法 前景{len(boxes_diff)}块", (0, 200, 255))
        ann_mog2 = annotate(frame, boxes_mog2, f"B MOG2 前景{len(boxes_mog2)}块", (0, 255, 0))

        if idx >= 1:  # 第 0 帧没有 diff 结果，跳过写入以免误导
            vw_diff.write(ann_diff)
            vw_mog2.write(ann_mog2)
            sep = np.full((h, 10, 3), 255, np.uint8)
            vw_cmp.write(np.hstack([ann_diff, sep, ann_mog2]))

        diff_maxarea, diff_fill = blob_metrics(boxes_diff)
        mog2_maxarea, mog2_fill = blob_metrics(boxes_mog2)
        stats.append(dict(
            frame=idx,
            framediff_ratio=mask_diff.sum() / 255 / (w * h),
            mog2_ratio=mask_mog2.sum() / 255 / (w * h),
            framediff_boxes=len(boxes_diff),
            mog2_boxes=len(boxes_mog2),
            framediff_maxarea=diff_maxarea,
            mog2_maxarea=mog2_maxarea,
            framediff_fill=diff_fill,
            mog2_fill=mog2_fill,
        ))

        if idx in (int(total * 0.25), int(total * 0.5), int(total * 0.75)):
            key_frames.append((idx, ann_diff, ann_mog2, mask_diff, mask_mog2))

        idx += 1

    cap.release()
    vw_diff.release()
    vw_mog2.release()
    vw_cmp.release()

    csv_path = OUT / f"q4_{stem}_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    valid = stats[1:] if len(stats) > 1 else stats
    # 口径说明：前景占比 / 有检出帧占比 面向全部帧；
    # 团块数、面积、填充率只在「检测到团块」的帧上统计 ——
    # 空帧按 0 计入会拉低均值，掩盖帧差法「空心」与 MOG2「实心」的真实差异
    fd = [s for s in valid if s["framediff_maxarea"] > 0]
    mg = [s for s in valid if s["mog2_maxarea"] > 0]
    n_all = len(stats)

    print(f"\n[统计] 共处理 {idx} 帧")
    print(f"{'':<22}{'帧差法':>12}{'MOG2':>12}")
    print("-" * 46)
    print(f"{'前景占比均值':<22}"
          f"{np.mean([s['framediff_ratio'] for s in valid]) * 100:>11.2f}%"
          f"{np.mean([s['mog2_ratio'] for s in valid]) * 100:>11.2f}%")
    print(f"{'前景占比（最大帧）':<22}"
          f"{max(s['framediff_ratio'] for s in valid) * 100:>11.2f}%"
          f"{max(s['mog2_ratio'] for s in valid) * 100:>11.2f}%")
    print(f"{'有检出的帧占比':<22}"
          f"{len(fd) / n_all * 100:>11.1f}%"
          f"{len(mg) / n_all * 100:>11.1f}%")
    print(f"{'平均团块数':<22}"
          f"{np.mean([s['framediff_boxes'] for s in fd]):>12.2f}"
          f"{np.mean([s['mog2_boxes'] for s in mg]):>12.2f}")
    print(f"{'最大团块平均面积(px²)':<22}"
          f"{np.mean([s['framediff_maxarea'] for s in fd]):>12.0f}"
          f"{np.mean([s['mog2_maxarea'] for s in mg]):>12.0f}")
    print(f"{'平均填充率(实心度)':<22}"
          f"{np.mean([s['framediff_fill'] for s in fd]):>12.3f}"
          f"{np.mean([s['mog2_fill'] for s in mg]):>12.3f}")
    print("-" * 46)
    print(f"[保存] {csv_path}")

    if key_frames:
        imgs, titles = [], []
        for fidx, ad, am, md, mm in key_frames:
            imgs += [ad, md, am, mm]
            titles += [f"第{fidx}帧 A帧差法", f"第{fidx}帧 A掩码",
                       f"第{fidx}帧 B MOG2", f"第{fidx}帧 B掩码"]
        save_grid(imgs, titles, "q4_frames.png", cols=4, figsize=(19, 3.6 * len(key_frames)),
                  suptitle="题四：帧差法 vs MOG2 背景减除（关键帧对比）")

    print(f"[保存] 视频已写入 {VIDEO_OUT}")
    for p in sorted(VIDEO_OUT.glob(f"q4_{stem}_*.mp4")):
        print(f"       {p.name}  {p.stat().st_size / 1024:.0f} KB")

    return dict(
        frames=idx,
        framediff_ratio=float(np.mean([s["framediff_ratio"] for s in valid])),
        mog2_ratio=float(np.mean([s["mog2_ratio"] for s in valid])),
        framediff_hit=len(fd) / len(valid),
        mog2_hit=len(mg) / len(valid),
        framediff_maxarea=float(np.mean([s["framediff_maxarea"] for s in fd])),
        mog2_maxarea=float(np.mean([s["mog2_maxarea"] for s in mg])),
        framediff_fill=float(np.mean([s["framediff_fill"] for s in fd])),
        mog2_fill=float(np.mean([s["mog2_fill"] for s in mg])),
    )


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
