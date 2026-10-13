import cv2
import numpy as np

from common import DATA


def make_image(path, size=(720, 1280)):
    h, w = size
    img = np.full((h, w, 3), 240, np.uint8)  # 浅灰背景 (BGR)

    # 彩色几何体：颜色分明，便于观察 B/G/R 三通道分离差异
    cv2.circle(img, (250, 250), 120, (60, 76, 231), -1)              # 红
    cv2.rectangle(img, (470, 130), (760, 370), (233, 180, 86), -1)   # 蓝
    cv2.circle(img, (990, 250), 115, (71, 160, 67), -1)              # 绿

    tri = np.array([[180, 560], [420, 560], [300, 400]], np.int32)
    cv2.fillPoly(img, [tri], (180, 105, 255))                        # 品红

    cv2.circle(img, (700, 500), 90, (30, 200, 240), -1)              # 黄
    cv2.rectangle(img, (900, 420), (1150, 620), (200, 120, 60), -1)  # 棕

    # 文字：验证通道分离后笔画清晰度
    cv2.putText(img, "ComSen 2026", (430, 690),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 40), 3, cv2.LINE_AA)

    cv2.imwrite(str(path), img)
    print(f"[生成] {path}  ({w}x{h})")
    return path


def _open_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频 {path}")
    return writer


def make_motion_video(path, size=(480, 640), fps=25, seconds=6):
    """固定背景 + 水平往返运动方块：经典的运动目标检测场景。"""
    w, h = size
    writer = _open_writer(path, fps, size)
    frames = fps * seconds

    rng = np.random.default_rng(42)
    # 静止背景：带轻微纹理，模拟真实摄像头底噪
    bg = np.full((h, w, 3), 60, np.uint8)
    for _ in range(120):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(4, 14)),
                   tuple(int(v) for v in rng.integers(50, 90, 3)), -1)

    for i in range(frames):
        frame = bg.copy()
        t = i / frames
        # 三角波：左->右->左
        x = int(w * (1 - abs(2 * t - 1)) * 0.7) + int(w * 0.15)
        y = h // 2 + int(60 * np.sin(4 * np.pi * t))
        cv2.rectangle(frame, (x - 45, y - 45), (x + 45, y + 45), (0, 220, 255), -1)
        # 轻微噪声，抬高检测难度
        noise = rng.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"[生成] {path}  ({w}x{h}, {seconds}s @ {fps}fps)")
    return path


def make_ball_video(path, size=(480, 640), fps=25, seconds=8):
    """小球沿正弦轨迹运动 + 十字纹理，部分帧被遮挡，用于追踪器实验。"""
    w, h = size
    writer = _open_writer(path, fps, size)
    frames = fps * seconds
    rng = np.random.default_rng(7)

    bg = np.full((h, w, 3), 45, np.uint8)
    for _ in range(200):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(2, 6)),
                   tuple(int(v) for v in rng.integers(35, 70, 3)), -1)

    radius = 22
    for i in range(frames):
        frame = bg.copy()
        t = i / (frames - 1)
        x = int(w * 0.1 + w * 0.8 * t)
        y = int(h / 2 + h * 0.32 * np.sin(2 * np.pi * 1.5 * t))
        # 小球：红橙实心 + 白色十字，给追踪器提供角点特征
        cv2.circle(frame, (x, y), radius, (30, 60, 235), -1)
        cv2.line(frame, (x - radius // 2, y), (x + radius // 2, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y - radius // 2), (x, y + radius // 2), (255, 255, 255), 2)

        # 每隔一段插入遮挡物，模拟真实场景中的遮挡
        if (i // (fps // 2)) % 3 == 2:
            cv2.rectangle(frame, (int(w * 0.45), int(h * 0.25)),
                          (int(w * 0.62), int(h * 0.8)), (120, 120, 120), -1)

        writer.write(frame)

    writer.release()
    print(f"[生成] {path}  ({w}x{h}, {seconds}s @ {fps}fps)")
    return path


def main():
    make_image(DATA / "sample_image.jpg")
    make_motion_video(DATA / "sample_motion.mp4")
    make_ball_video(DATA / "sample_ball.mp4")
    print(f"\n素材已写入 {DATA}")
    print("提示：真实 example 素材放入同目录后，各脚本会自动优先使用它们。")


if __name__ == "__main__":
    main()
