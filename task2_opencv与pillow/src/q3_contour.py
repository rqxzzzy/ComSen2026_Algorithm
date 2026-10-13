import cv2
import numpy as np

from common import resolve_image, save_grid
from common import OUT

MIN_AREA = 200  # 小于该面积的连通域视为噪点丢弃


def annotate(img, contours):
    """在图像上绘制轮廓、编号，返回绘制结果和各轮廓明细。"""
    canvas = img.copy()
    detail = []

    for idx, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        cv2.drawContours(canvas, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(canvas, str(idx), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        detail.append(dict(idx=idx, area=area, perimeter=peri, center=(cx, cy)))

    return canvas, detail


def main():
    path = resolve_image("example1.jpg")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取 {path}")
    h, w = img.shape[:2]
    print(f"[读取] {path.name}  {w}x{h}")

    # 预处理：去噪、反相二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 轮廓提取与筛选
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    raw_count = len(contours)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    contours.sort(key=cv2.contourArea, reverse=True)

    print(f"[检测] 得到 {raw_count} 个外轮廓，"
          f"过滤面积 < {MIN_AREA} 后保留 {len(contours)} 个")

    canvas, detail = annotate(img, contours)

    total_area = sum(d["area"] for d in detail)
    total_perimeter = sum(d["perimeter"] for d in detail)

    print("\n[明细]")
    print(f"{'编号':<6}{'面积(px²)':>14}{'周长(px)':>14}")
    for d in detail:
        print(f"{d['idx']:<6}{d['area']:>14.2f}{d['perimeter']:>14.2f}")
    print("-" * 40)
    print(f"{'合计':<6}{total_area:>14.2f}{total_perimeter:>14.2f}")

    out_path = OUT / "q3_contours.jpg"
    cv2.imwrite(str(out_path), canvas)
    print(f"\n[保存] {out_path}")

    save_grid(
        [img, binary, canvas],
        ["原图", "二值化（Otsu 反相）",
         f"轮廓检测结果\n总轮廓数={len(contours)}  "
         f"总面积={total_area:.0f} px²  总周长={total_perimeter:.0f} px"],
        "q3_out.png",
        cols=3,
        figsize=(17, 6),
        suptitle="题三：轮廓检测与面积/周长统计",
    )

    return total_area, total_perimeter


if __name__ == "__main__":
    main()
