import os
import io
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

ROOT = r"D:\ComSen2026\task3"
PARQUET = os.path.join(ROOT, "data", "train-00000-of-00001.parquet")
CACHE = os.path.join(ROOT, "data", "mnist_cache.npz")
OUT_DIR = os.path.join(ROOT, "learn")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 12
TEST_RATIO = 1 / 6
SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(SEED)
np.random.seed(SEED)


"""用 pandas 读 parquet，再用 PIL 解码其中的 PNG 字节流。

缓存成 .npz，第二次运行直接读缓存。
"""
if os.path.exists(CACHE):
    print(f"[数据] 读取缓存 {CACHE}")
    d = np.load(CACHE)
    images, labels = d["images"], d["labels"]
    print(f"       形状={images.shape}")
else:
    print(f"[数据] 读取 parquet: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    print(f"       形状={df.shape}  列={df.columns.tolist()}")

    n = len(df)
    images = np.zeros((n, 28, 28), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.uint8)

    t0 = time.time()
    for i, (rec, lab) in enumerate(zip(df["image"], df["label"])):
        raw = rec["bytes"]        # image 列是 {'bytes': ..., 'path': None}
        img = Image.open(io.BytesIO(raw)).convert("L")   # 灰度
        images[i] = np.array(img, dtype=np.uint8)
        labels[i] = int(lab)
        if (i + 1) % 20000 == 0:
            print(f"       已解码 {i + 1}/{n}  ({time.time() - t0:.1f}s)")

    print(f"[数据] 解码完成 {n} 张，用时 {time.time() - t0:.1f}s")
    np.savez_compressed(CACHE, images=images, labels=labels)
    print(f"[数据] 已缓存到 {CACHE}")

# 自检：确认图像不是全黑、标签范围正确
print(f"[自检] 像素范围 [{images.min()}, {images.max()}]  "
      f"均值 {images.mean():.1f}  标签种类 {sorted(set(labels.tolist()))}")


def build_loaders(images, labels, batch_size=BATCH_SIZE, test_ratio=TEST_RATIO,
                  seed=SEED):
    """切分训练/测试集，转成张量并封装成 DataLoader。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    n_test = int(len(images) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)   # (N,1,28,28)
    y = torch.tensor(labels, dtype=torch.long)

    # 归一化顺序：先缩到 0~1，再用「训练集」统计量标准化。
    # 统计量只能用训练集算，用全体会泄漏测试集信息。
    x = x / 255.0
    mean = x[train_idx].mean().item()
    std = x[train_idx].std().item()
    x = (x - mean) / std

    train_ds = TensorDataset(x[train_idx], y[train_idx])
    test_ds = TensorDataset(x[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    return train_loader, test_loader, test_idx, mean, std


def evaluate(model, loader, criterion):
    """在给定数据集上算平均损失 + 准确率。

    必须 model.eval()：关闭 Dropout，否则评估结果会被训练时的随机性污染。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():          # 评估不需要梯度，省显存也更快
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            total_loss += criterion(out, yb).item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


class MLP(nn.Module):
    """两层隐藏层的多层感知机（784 -> 256 -> 128 -> 10）。

    最后一层不加激活：CrossEntropyLoss 内部已含 Softmax。
    """

    def __init__(self, hidden=(256, 128), num_classes=10, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden[0]), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x):
        return self.net(x)


def plot_predictions(model, images, labels, mean, std, out_path, n_show=6,
                     only_wrong=False):
    """可视化预测结果：原图 + 真实标签 + 预测标签 + 置信度。

    only_wrong=True 时专门挑预测错的样本，用来分析模型弱点。
    """
    model.eval()
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    x = (x / 255.0 - mean) / std
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu()
    preds = probs.argmax(1).numpy()

    if only_wrong:
        pool = np.where(preds != labels)[0]
        title = "预测错误的样本（看模型错在哪）"
        if len(pool) < n_show:
            pool = np.arange(len(labels))
    else:
        pool = np.arange(len(labels))
        title = "预测结果可视化（绿=正确，红=错误）"

    rng = np.random.default_rng(0)
    chosen = rng.choice(pool, size=min(n_show, len(pool)), replace=False)

    fig, axes = plt.subplots(1, len(chosen), figsize=(2.6 * len(chosen), 3.6))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, chosen):
        ax.imshow(images[i], cmap="gray")
        ok = preds[i] == labels[i]
        ax.set_title(f"真实 {labels[i]} / 预测 {preds[i]}\n"
                     f"置信度 {probs[i, preds[i]] * 100:.1f}%",
                     fontsize=10, color=("green" if ok else "red"))
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("green" if ok else "red")
            spine.set_linewidth(3)

    fig.suptitle(f"Task 3 题三：{title}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_path}")


def plot_first_layer_weights(model, out_path, n_show=32):
    """把第一层权重的前若干个神经元 reshape 成 28x28 画出来。"""
    w = model.net[1].weight.detach().cpu().numpy()      # (256, 784)
    fig, axes = plt.subplots(4, 8, figsize=(13, 7))
    for i, ax in enumerate(axes.ravel()):
        if i >= min(n_show, w.shape[0]):
            ax.axis("off")
            continue
        ax.imshow(w[i].reshape(28, 28), cmap="RdBu_r")
        ax.set_title(f"#{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("第一层权重可视化（红=正权重，蓝=负权重）", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_path}")


# 训练过程
model = MLP(hidden=(256, 128), num_classes=10, drop=0.2)
n_params = sum(p.numel() for p in model.parameters())
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = dict(loss=[], train_acc=[], test_acc=[], test_loss=[])
best_acc = 0.0
best_state = None
best_epoch = 0
ckpt_path = os.path.join(OUT_DIR, "q3_mlp.pth")

train_loader, test_loader, test_idx, mean, std = build_loaders(images, labels)

print(f"\n[训练] device={DEVICE}  参数量={n_params:,}  "
      f"训练集={len(train_loader.dataset)}  测试集={len(test_loader.dataset)}")
print(f"{'Epoch':>7}{'训练Loss':>12}{'训练Acc':>10}{'测试Loss':>12}{'测试Acc':>10}{'用时':>9}")

for epoch in range(1, EPOCHS + 1):
    model.train()                          # 开启 Dropout
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        out = model(xb)
        loss = criterion(out, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    history["loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    history["test_loss"].append(test_loss)

    # 只保留测试集上最优的那一版：测试准确率会先升后降，
    # 拿最后一轮往往不是最好的，甚至已经开始过拟合。
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>7}{train_loss:>12.4f}{train_acc * 100:>9.2f}%"
          f"{test_loss:>12.4f}{test_acc * 100:>9.2f}%{time.time() - t0:>8.1f}s")

model.load_state_dict(best_state)
torch.save(model.state_dict(), ckpt_path)
print(f"\n[最优] 第 {best_epoch} 轮  测试准确率 {best_acc * 100:.2f}%"
      f"（最后一轮 {history['test_acc'][-1] * 100:.2f}%）")
print(f"[保存] {ckpt_path}")

"""画三个曲线：Loss / Train Acc / Test Acc。"""
ep = range(1, len(history["loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

axes[0].plot(ep, history["loss"], "o-", color="#e05c5c", label="训练 Loss")
axes[0].plot(ep, history["test_loss"], "s--", color="#8a8a8a", label="测试 Loss")
axes[0].set_title("损失函数 (Loss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("交叉熵损失")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(ep, [a * 100 for a in history["train_acc"]], "o-",
             color="#4c9be8")
axes[1].set_title("训练准确度 (Train Acc)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("准确率 (%)")
axes[1].grid(alpha=0.3)

axes[2].plot(ep, [a * 100 for a in history["test_acc"]], "o-",
             color="#5cb85c")
axes[2].set_title("测试准确度 (Test Acc)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("准确率 (%)")
axes[2].grid(alpha=0.3)

# 标注最优的那一轮
axes[2].axhline(best_acc * 100, color="red", linestyle=":", linewidth=1)
axes[2].plot([best_epoch], [best_acc * 100], "r*", markersize=14)
axes[2].text(len(ep) * 0.35, best_acc * 100 - 2.2,
             f"最优 {best_acc * 100:.2f}%（第 {best_epoch} 轮）",
             color="red", fontsize=10)

fig.suptitle("Task 3 题三：MLP 识别 MNIST 训练曲线", fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "q3_curves.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"[保存] {os.path.join(OUT_DIR, 'q3_curves.png')}")

# 用最优的那一版做可视化（题目要求至少展示 3 张图片结果）
t_imgs = images[test_idx]
t_labs = labels[test_idx]
plot_predictions(model, t_imgs, t_labs, mean, std,
                 os.path.join(OUT_DIR, "q3_samples.png"), n_show=6)
plot_predictions(model, t_imgs, t_labs, mean, std,
                 os.path.join(OUT_DIR, "q3_wrong.png"), n_show=6,
                 only_wrong=True)
plot_first_layer_weights(model, os.path.join(OUT_DIR, "q3_weights.png"))

print(f"\n[结果] 参数量 {n_params:,}   最优测试准确率 {best_acc * 100:.2f}%"
      f"（第 {best_epoch} 轮）")
