import os
import shutil
import sys

from ultralytics import YOLO

ROOT = r"D:\ComSen2026\task4"
DATASET_DIR = os.path.join(ROOT, "datasets", "drone_bird")
WEIGHTS = os.path.join(ROOT, "weights")
RUNS = os.path.join(ROOT, "runs")

EPOCHS = 60
IMGSZ = 640          # YOLO 标准输入尺寸
BATCH = 16
WORKERS = 4
BASE_MODEL = "yolo11n.pt"
RUN_NAME = "drone_yolo11n"
amp = False


def ensure_dataset_yaml(yaml_path):
    """Roboflow 导出的 data.yaml 里可能是相对路径，数据集目录一移动就失效。

    不硬拼路径，而是按候选列表挑「真实存在」的那个，修正后另存为
    data_fixed.yaml（不动原始文件）。
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(yaml_path)
    # 注意：yaml 里写的是 val，但 Roboflow 导出的目录叫 valid —— 两种都要试
    DIR_NAMES = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
    changed = False
    for split in ("train", "val", "test"):
        raw = cfg.get(split)
        if not raw:
            continue
        candidates = [str(raw)]                              # 原样
        for d in DIR_NAMES[split]:
            candidates.append(os.path.join(root, d, "images"))   # 标准布局
        candidates.append(os.path.join(root, os.path.basename(str(raw).rstrip("/\\"))))
        for c in candidates:
            if c and os.path.isdir(c):
                if c != str(raw):
                    cfg[split] = c.replace("\\", "/")
                    changed = True
                break

    if not changed:
        return yaml_path

    fixed = os.path.join(root, "data_fixed.yaml")
    with open(fixed, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[数据集] 路径已修正 -> {fixed}")
    return fixed


def main():
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[错误] 在 {DATASET_DIR} 找不到 data.yaml，请先解压数据集")
        sys.exit(1)

    yaml_path = ensure_dataset_yaml(yaml_path)

    print(f"[数据集] {yaml_path}")


    for split in ["train", "valid", "test"]:
        d = os.path.join(os.path.dirname(yaml_path), split, "images")
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            print(f"[数据集] {split}: {n} 张")

    base = os.path.join(WEIGHTS, BASE_MODEL)
    if not os.path.exists(base):
        base = BASE_MODEL       # 交给 ultralytics 自己找/下载
    print(f"[模型] 基础权重 {base}")

    model = YOLO(base)

    print(f"[训练] epochs={EPOCHS} imgsz={IMGSZ} batch={BATCH} ...")
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=RUNS,
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        amp=amp,             # ← 必须显式传！否则 ultralytics 会做 AMP 检查并
                             #    去 github 下载 yolo11n.pt，本机网络下会一直卡住
        plots=True,          # 生成混淆矩阵 / results.png 等评估图
        patience=20,         # 早停
        seed=42,
        verbose=True,
    )

    # 显式传 project 时 Ultralytics 不再补 detect/ 这一层，两个候选路径都试一下
    best = None
    for cand in [
        os.path.join(RUNS, RUN_NAME, "weights", "best.pt"),
        os.path.join(RUNS, "detect", RUN_NAME, "weights", "best.pt"),
    ]:
        if os.path.exists(cand):
            best = cand
            break

    if best:
        dst = os.path.join(WEIGHTS, "drone.pt")
        shutil.copy2(best, dst)
        print(f"\n[导出] {best}\n   ->  {dst}")
    else:
        print(f"\n[警告] 没找到 best.pt，请检查 {RUNS}")

    print(f"[完成] 训练结果目录: {os.path.join(RUNS, RUN_NAME)}")
    return results


if __name__ == "__main__":
    main()
