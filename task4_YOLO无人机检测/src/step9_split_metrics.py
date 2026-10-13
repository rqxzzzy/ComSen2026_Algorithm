import os
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = r"D:\ComSen2026\task4"
WEIGHTS = os.path.join(ROOT, "weights")
DATA_YAML = os.path.join(ROOT, "datasets", "drone_bird", "data.yaml")
OUT = os.path.join(ROOT, "outputs")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 对训练好的模型在 train / val / test 三个划分上分别评估
MODELS = [("本模型", os.path.join(WEIGHTS, "drone.pt"))]


def main():
    rows = []
    for tag, path in MODELS:
        if not os.path.exists(path):
            print(f"[跳过] {path} 不存在")
            continue
        print("=" * 68)
        print(f"[{tag}] {os.path.basename(path)}")
        print("=" * 68)
        m = YOLO(path)

        for split in ["train", "val", "test"]:
            try:
                r = m.val(data=DATA_YAML, split=split, imgsz=640,
                          verbose=False, plots=False,
                          project=os.path.join(ROOT, "runs_split"),
                          name=f"{tag}_{split}")
                rec = {
                    "model": tag, "split": split,
                    "images": len(r.speed) and int(r.box.nc and 0) or 0,
                    "P": float(r.box.mp), "R": float(r.box.mr),
                    "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                }
                names = m.names
                for i, c in enumerate(r.box.ap_class_index):
                    rec[f"mAP50_{names[int(c)]}"] = float(r.box.ap50[i])
                rows.append(rec)
                print(f"  [{split:<5}] P={rec['P']:.4f}  R={rec['R']:.4f}  "
                      f"mAP50={rec['mAP50']:.4f}  mAP50-95={rec['mAP50_95']:.4f}")
            except Exception as e:
                print(f"  [{split}] 失败: {e}")

    if not rows:
        print("没有结果")
        return

    keys = sorted({k for r in rows for k in r})
    order = ["model", "split", "P", "R", "mAP50", "mAP50_95"]
    cols = order + [k for k in keys if k not in order]
    csv_path = os.path.join(OUT, "split_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[保存] {csv_path}")

    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, metric in zip(axes, ["P", "R", "mAP50", "mAP50_95"]):
        x = np.arange(len(splits))
        w = 0.36
        for k, (tag, _) in enumerate(MODELS):
            vals = []
            for s in splits:
                hit = [r for r in rows if r["model"] == tag and r["split"] == s]
                vals.append(hit[0][metric] if hit else 0)
            bars = ax.bar(x + (k - 0.5) * w, vals, w,
                          label=f"{tag}（drone.pt）",
                          color=["#55a868", "#c44e52"][k % 2])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["训练集\n(train)", "验证集\n(val)", "测试集\n(test)"])
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)

    fig.suptitle("Task 4 方向1：三个数据集划分上的检测指标"
                 "（YOLO 无「准确度」，用 P/R/mAP 对应题目的 Train Acc / Test Acc）",
                 fontsize=14)
    fig.tight_layout()
    p = os.path.join(OUT, "split_metrics.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {p}")


if __name__ == "__main__":
    main()
