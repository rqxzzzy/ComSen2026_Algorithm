import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_sample  # noqa: E402
import q1_channels  # noqa: E402
import q2_transform  # noqa: E402
import q3_contour  # noqa: E402
import q4_motion  # noqa: E402
import q5_tracker  # noqa: E402
from common import DATA  # noqa: E402


def banner(step, title):
    print("\n" + "=" * 72)
    print(f"  [{step}] {title}")
    print("=" * 72)


def main():
    banner("0/6", "素材检查")
    real = sorted(DATA.glob("example*"))
    if real:
        print(f"  发现真实素材：{[p.name for p in real]}")
    else:
        print("  未发现 example 素材，生成替代素材以跑通流程")
        make_sample.main()

    banner("1/6", "题一：通道分离")
    q1_channels.main()

    banner("2/6", "题二：缩放 / 旋转 / 平移 / 翻转")
    q2_transform.main()

    banner("3/6", "题三：轮廓检测与面积/周长统计")
    q3_contour.main()

    banner("4/6", "题四：MOG2 背景减除 vs 帧差法")
    q4_motion.main()

    banner("5/6", "题五：tracker 追踪小球")
    # 题目要求对 example3.mp4 与 example4.mp4 分别追踪，各自输出独立结果
    targets = [n for n in ("example3.mp4", "example4.mp4") if (DATA / n).exists()]
    if not targets:
        targets = ["example3.mp4"]  # 无真实素材时用替代素材跑一遍
    for name in targets:
        print(f"\n----- 追踪目标：{name} -----")
        q5_tracker.run(name)

    banner("6/6", "全部完成")
    from common import OUT
    print(f"  结果目录：{OUT}")


if __name__ == "__main__":
    main()
