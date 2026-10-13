import os
import time

from ultralytics import YOLO

ROOT = r"D:\ComSen2026\task4"
EXAMPLE_DIR = os.path.join(ROOT, "data", "example")
WEIGHTS = os.path.join(ROOT, "weights")
OUT_DIR = os.path.join(ROOT, "outputs")

# 官方预训练权重（COCO 80 类）。n=最小最快，s=小号精度更高
MODELS = ["yolo11n.pt", "yolo11s.pt"]
IMAGES = ["coco-bear.jpg", "coco-bus.jpg", "coco-person.jpg"]

os.makedirs(OUT_DIR, exist_ok=True)

for m in MODELS:
    wpath = os.path.join(WEIGHTS, m)
    if not os.path.exists(wpath):
        print(f"[跳过] 找不到权重 {wpath}")
        continue

    print(f"\n{'=' * 70}\n[模型] {m}\n{'=' * 70}")
    model = YOLO(wpath)          # 用新名字，别覆盖循环变量 m（下面还要用它拼文件名）

    for img in IMAGES:
        ipath = os.path.join(EXAMPLE_DIR, img)
        stem = os.path.splitext(img)[0]

        t0 = time.perf_counter()
        results = model.predict(ipath, conf=0.25, verbose=False)
        dt = time.perf_counter() - t0

        r = results[0]

        save_path = os.path.join(OUT_DIR, f"img_{stem}_{os.path.splitext(m)[0]}.jpg")
        r.save(filename=save_path)

        print(f"\n  [{img}]  推理耗时 {dt * 1000:.1f} ms")
        if r.boxes is None or len(r.boxes) == 0:
            print("    (未检测到目标)")
            continue
        for b in r.boxes:
            cls_id = int(b.cls.item())
            name = model.names[cls_id]
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            print(f"    {name:<15} 置信度 {conf:.3f}  框({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
        print(f"    -> 保存 {save_path}")
