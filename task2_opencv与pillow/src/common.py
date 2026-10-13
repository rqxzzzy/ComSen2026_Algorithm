from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境也能出图（服务器/后台运行）
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # .../task2
DATA = ROOT / "data"        # 输入素材
OUT = ROOT / "outputs"      # 输出图片
DOCS = ROOT / "docs"        # 报告
VIDEO_OUT = OUT / "videos"  # 输出视频

for _d in (DATA, OUT, DOCS, VIDEO_OUT):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def resolve_image(preferred: str = "example1.jpg") -> Path:
    """优先使用真实素材，缺失时回退到自生成的 sample_image.jpg。"""
    p = DATA / preferred
    if p.exists():
        return p
    fallback = DATA / "sample_image.jpg"
    if not fallback.exists():
        raise FileNotFoundError(
            f"缺少素材 {p}\n"
            f"请把 example 素材放到 {DATA}，或先运行 python src/make_sample.py 生成替代图。"
        )
    print(f"[提示] 未找到 {p.name}，改用 {fallback.name} 演示（放上真实素材后会自动切换）")
    return fallback


def resolve_video(preferred: str) -> Path:
    """视频素材回退：缺失时提示，由调用方决定是否用合成视频。"""
    p = DATA / preferred
    if p.exists():
        return p
    return None


def to_rgb(img):
    """OpenCV 的 BGR -> matplotlib 需要的 RGB。"""
    return img[:, :, ::-1] if img.ndim == 3 and img.shape[2] == 3 else img


def save_grid(images, titles, out_name, cols=3, figsize=None, cmap=None,
              suptitle=None, dpi=120):
    """把若干张图排成网格保存到 outputs/，返回保存路径。"""
    if not isinstance(images, (list, tuple)):
        images, titles = [images], [titles]
    n = len(images)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    figsize = figsize or (4.6 * cols, 4.0 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, img, title in zip(axes.ravel(), images, titles):
        if img.ndim == 3:
            ax.imshow(to_rgb(img))
        else:
            ax.imshow(img, cmap=cmap or "gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, y=1.0)
    fig.tight_layout()
    path = OUT / out_name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {path}")
    return path
