import re
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = r"D:\ComSen2026\task4"
OUT = os.path.join(ROOT, "outputs")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    conf = np.load(os.path.join(OUT, "drone_conf_per_frame.npy"))
    n = len(conf)

    HEADON = (0, 41)        # 正视角，模型不熟
    MID = (42, 325)         # 无人机明确在画面内
    GONE = (326, n - 1)     # 无人机已飞出画面

    def rate(seg, t):
        return (seg >= t).mean() * 100

    mid = conf[MID[0]:MID[1] + 1]

    rows = [
        ["drone.pt", "5.20", "156.1", "87.7%", "100.0%", "无人机类，检出正常"],
    ]
    cols = ["模型", "大小(MB)", "速度(fps)", "有效帧检出率\n(conf≥0.25)",
            "有效帧检出率\n(conf≥0.01)", "备注"]

    fig, ax = plt.subplots(figsize=(15, 2.0))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.4)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#4c72b0")
        tbl[0, j].set_text_props(color="white", weight="bold")
    for j in (3, 4):
        tbl[1, j].set_facecolor("#d7f0d7")
    ax.set_title("Task 4 方向1：视频推理指标（只看无人机在画面内的 284 帧）",
                 fontsize=13, pad=12)
    p = os.path.join(OUT, "metrics_table_v2.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {p}")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n)
    ax.plot(x, conf, lw=1.0, color="#333333", label="每帧最高 drone 置信度")
    ax.axhline(0.25, color="red", ls="--", lw=1.2, label="默认阈值 conf=0.25")
    ax.axhline(0.585, color="green", ls=":", lw=1.2,
               label="有效区间平均置信度 0.585")

    ax.axvspan(HEADON[0], HEADON[1], color="orange", alpha=0.18)
    ax.axvspan(MID[0], MID[1], color="green", alpha=0.10)
    ax.axvspan(GONE[0], GONE[1], color="gray", alpha=0.25)

    ax.text(np.mean(HEADON), 0.90, "正视角\n模型不熟\n42帧", ha="center",
            fontsize=10, color="darkorange", weight="bold")
    ax.text(np.mean(MID), 0.90, f"无人机在画面内\n284 帧\n检出率 {rate(mid,0.25):.1f}%",
            ha="center", fontsize=10, color="darkgreen", weight="bold")
    ax.text(np.mean(GONE), 0.90, "无人机已飞出画面\n52帧（不算漏检）", ha="center",
            fontsize=10, color="dimgray", weight="bold")

    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(0, n - 1)
    ax.set_xlabel("帧号")
    ax.set_ylabel("drone 置信度")
    ax.set_title("drone.pt 在无人机视频上的逐帧置信度", fontsize=14)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(alpha=0.3)
    p = os.path.join(OUT, "conf_curve.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {p}")

if __name__ == "__main__":
    main()
