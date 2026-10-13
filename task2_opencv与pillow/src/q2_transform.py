import cv2
import numpy as np

from common import resolve_image, save_grid

BORDER = (240, 240, 240)  # 变换后露出的画布颜色，与示例图底色一致


def main():
    path = resolve_image("example1.jpg")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取 {path}")
    h, w = img.shape[:2]
    print(f"[读取] {path.name}  尺寸={w}x{h}")

    # 旋转 45°：先把画布放大到能装下旋转后的外接矩形，再平移回中心
    center = (w / 2, h / 2)
    M_1 = cv2.getRotationMatrix2D(center, 45, 1.0)
    cos, sin = abs(M_1[0, 0]), abs(M_1[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M_1[0, 2] += new_w / 2 - center[0]
    M_1[1, 2] += new_h / 2 - center[1]

    M_2 = np.float32([[1, 0, 150], [0, 1, 180]])

    img_list = [
        (cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR), "缩放 0.5x"),
        (cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR), "缩放 1.5x"),
        (cv2.warpAffine(img, M_1, (new_w, new_h), borderValue=BORDER), "旋转 45°（画布自适应）"),
        (cv2.warpAffine(img, M_2, (w, h), borderValue=BORDER), "平移 (+150, +80)"),
        (cv2.flip(img, 1), "水平翻转"),
        (cv2.flip(img, 0), "垂直翻转"),
        (cv2.flip(img, -1), "水平 + 垂直翻转"),
    ]

    print("\n[输出] 各变换结果尺寸")
    for out, name in img_list:
        print(f"       {name:<22} {out.shape[1]}x{out.shape[0]}")

    save_grid(
        [img] + [i[0] for i in img_list],
        ["原图"] + [i[1] for i in img_list],
        "q2_out.png",
        cols=3,
        figsize=(15, 13),
        suptitle="题二：缩放 / 旋转 / 平移 / 翻转",
    )


if __name__ == "__main__":
    main()
