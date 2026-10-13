import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = r"D:\ComSen2026\task3\learn\q2_fit.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 真值系数：仅用于最后对比检验，训练过程中绝不使用
TRUE = {"a1": 2.0, "a2": 1.5, "a3": 3.0, "a4": 0.5,
        "a5": -0.5, "a6": 0.8, "b": 2026.0}

PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6", "b"]


def true_function(x1, x2, p=TRUE):
    """真实函数（不含噪声）。x1/x2 为 numpy 数组。"""
    return (p["a1"] * x1 ** 2
            + p["a2"] * x1 * x2
            + p["a3"] * np.sin(x1)
            + p["a4"] * np.cos(2 * x2)
            + np.exp(p["a5"] * x1)
            + p["a6"] * x2
            + p["b"])


def make_data(n=3000, lo=-3.0, hi=3.0, noise=3.0, seed=42):
    """随机采样 n 个 (x1, x2)，返回 (x1, x2, y_clean, y_obs)。

    y_clean 不含噪声（画真实曲面、评估学到的函数多准），y_obs 才是训练输入。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    y_clean = true_function(x1, x2)
    y_obs = y_clean + rng.uniform(-noise, noise, n)
    return x1, x2, y_clean, y_obs


def build_params(b_init=0.0):
    """把 7 个参数建成 nn.Parameter —— PyTorch 才会追踪它们的梯度。"""
    return {
        "a1": nn.Parameter(torch.tensor(0.0)),
        "a2": nn.Parameter(torch.tensor(0.0)),
        "a3": nn.Parameter(torch.tensor(0.0)),
        "a4": nn.Parameter(torch.tensor(0.0)),
        "a5": nn.Parameter(torch.tensor(0.0)),
        "a6": nn.Parameter(torch.tensor(0.0)),
        "b": nn.Parameter(torch.tensor(float(b_init))),   # 聪明初始化：全 0 会让初始 MSE ~4e6
    }


def model(x1, x2, p):
    """参数化模型。注意 a5 在指数上，其余参数都是线性使用。"""
    return (p["a1"] * x1 ** 2
            + p["a2"] * x1 * x2
            + p["a3"] * torch.sin(x1)
            + p["a4"] * torch.cos(2 * x2)
            + torch.exp(p["a5"] * x1)
            + p["a6"] * x2
            + p["b"])


def train(x1, x2, y_obs, params, lr=0.05, epochs=8000, log_every=800):
    """梯度下降训练，返回每次记录的 loss 曲线。

    用 Adam：参数间量级差异极大（b~2026 而 a5~-0.5），需要自适应步长。
    """
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        y_pred = model(x1, x2, params)
        loss = loss_fn(y_pred, y_obs)

        # 三行心脏，顺序不能错。zero_grad 必须调用——PyTorch 梯度默认累加。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:>5}   loss = {loss.item():.6f}")

    return history


def r2_score(y_true, y_pred):
    """决定系数 R² = 1 - SS_res / SS_tot（SS_tot 以「预测平均值」为基线）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def plot_surfaces(x1, x2, y_clean, y_fit, r2_clean, r2_obs, out_path=OUT):
    """左右并排画两个 3D 曲面：真实函数 vs 拟合函数。

    用规则网格而不是训练用的随机散点，曲面才干净。
    """
    g = np.linspace(-3, 3, 120)
    G1, G2 = np.meshgrid(g, g)
    Z_true = true_function(G1, G2)
    Z_fit = y_fit

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(G1, G2, Z_true, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_title("真实函数（不含噪声）", fontsize=14)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(G1, G2, Z_fit, cmap="viridis", edgecolor="none", alpha=0.9)
    ax2.set_title(f"PyTorch 拟合函数\n"
                  f"R²(对真值)={r2_clean:.5f}   R²(对含噪观测)={r2_obs:.5f}",
                  fontsize=13)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    # 两个子图 z 轴范围必须一致，否则视觉上没法比较
    zmin, zmax = Z_true.min(), Z_true.max()
    ax1.set_zlim(zmin, zmax)
    ax2.set_zlim(zmin, zmax)

    fig.suptitle("Task 3 题二：梯度下降拟合原函数", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_path}")


def plot_loss(history, out_path):
    """画一下 loss 曲线，直观看到收敛过程。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history, linewidth=1.2)
    ax.set_yscale("log")          # 损失跨了几个数量级，用对数轴才看得清
    ax.set_xlabel("迭代次数 (epoch)")
    ax.set_ylabel("MSE 损失（对数轴）")
    ax.set_title("训练损失曲线")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_path}")


def main():
    x1_np, x2_np, y_clean, y_obs = make_data(n=3000)
    print(f"[数据] 样本数={len(x1_np)}  "
          f"y 范围=[{y_obs.min():.1f}, {y_obs.max():.1f}]  "
          f"y 均值={y_obs.mean():.1f}")
    print(f"[说明] 若把参数全初始化为 0，首帧预测只有 1.0，"
          f"初始 MSE 会高达 {(1 - y_obs.mean()) ** 2:.2e}")
    print(f"[初始化] 因此把 b 初始化为 y 的均值 {y_obs.mean():.1f}\n")

    X1 = torch.tensor(x1_np, dtype=torch.float32)
    X2 = torch.tensor(x2_np, dtype=torch.float32)
    Y = torch.tensor(y_obs, dtype=torch.float32)

    params = build_params(b_init=float(y_obs.mean()))
    print("[训练] 开始梯度下降 ...")
    history = train(X1, X2, Y, params, lr=0.05, epochs=8000)

    fitted = {k: float(v.detach()) for k, v in params.items()}
    print(f"\n{'参数':<6}{'真值':>12}{'拟合值':>14}{'绝对误差':>14}")
    print("-" * 48)
    for k in PARAM_NAMES:
        print(f"{k:<6}{TRUE[k]:>12.4f}{fitted[k]:>14.4f}"
              f"{abs(fitted[k] - TRUE[k]):>14.4f}")

    y_pred = model(X1, X2, params).detach().numpy()
    r2_obs = r2_score(y_obs, y_pred)
    r2_clean = r2_score(y_clean, y_pred)
    print("-" * 48)
    print(f"R²（对含噪观测值） = {r2_obs:.6f}")
    print(f"R²（对真值）       = {r2_clean:.6f}")
    print("提示：含噪口径受噪声上限约束（噪声方差 3 / 真值方差约 50），"
          "约 0.94 已是该口径的天花板；对真值的口径更能反映函数学得准不准。")

    g = np.linspace(-3, 3, 120)
    G1, G2 = np.meshgrid(g, g)
    with torch.no_grad():
        Z_fit = model(torch.tensor(G1, dtype=torch.float32),
                      torch.tensor(G2, dtype=torch.float32),
                      params).numpy()
    plot_surfaces(G1, G2, y_clean, Z_fit, r2_clean, r2_obs)
    plot_loss(history, OUT.replace("q2_fit", "q2_loss"))

    return fitted, r2_obs, r2_clean


if __name__ == "__main__":
    main()
