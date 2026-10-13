import cv2
import numpy as np
from PIL import Image

from common import resolve_image, save_grid

path = resolve_image("example1.jpg")

img_cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
if img_cv is None:
    raise IOError(f"OpenCV 无法读取 {path}")

with Image.open(path) as im:
    rgb = np.array(im.convert("RGB"))
img_pil = rgb[:, :, ::-1].copy()

print(f"[读取] {path.name}")
print(f"       OpenCV  shape={img_cv.shape}")
print(f"       Pillow  shape={img_pil.shape}")
print(f"[校验] 两个库读出的像素完全一致：{np.array_equal(img_cv, img_pil)}")

b, g, r = cv2.split(img_cv)

save_grid(
    [img_cv, b, g, r],
    ["原图（BGR）", "B 通道（灰度）", "G 通道（灰度）", "R 通道（灰度）"],
    "q1_out.png",
    cols=4,
    figsize=(17, 5),
    suptitle="题一：原图与三通道分离",
)
