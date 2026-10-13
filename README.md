# ComSen2026 算法组招新题 · 提交仓库

> **班别**：光电2606　**姓名**：饶乾鑫
> 提交日期：2026-10-17

本仓库为 ComSen2026 算法组招新题（CV / 深度学习方向）的完成成果，包含 Task 1 ~ Task 4 全部内容。

---

## 目录结构

```
.
├── README.md                    本文件（含 Task 1 环境配置说明）
├── task2_opencv与pillow/          Task 2：OpenCV 与 Pillow 图像处理
│   └── src/                      5 题实现（q1 通道分离 / q2 几何变换 / q3 二值化与轮廓 / q4 运动检测 / q5 追踪器对比）
├── task3_MLP识别MNIST/            Task 3：神经网络与 MLP 识别 MNIST
│   └── src/                      题二拟合 + 题三 MLP 完整代码
└── task4_YOLO无人机检测/          Task 4（自主选题方向一）：YOLO 目标检测与量化
    └── src/                      10 个流程脚本 + 数据集下载器
```

> **本仓库只放代码**：题目要求「将**除代码外**文件（如视频、图片、报告）打包发送到指定邮箱」，
> 因此实验报告、结果图、结果视频、模型权重等均不纳入本仓库，已随邮件一并提交。
> 数据集（MNIST parquet、drone-bird 图片）体积较大，可通过题目提供的链接重新获取。

> **Task 1 没有单独的目录**：它只要求搭建环境，没有代码或文档产出，因此环境说明直接写在本文件里（见下）。

---

## 任务速览

### Task 1 · 环境配置

在 Windows 11 上配置了可直接进行深度学习训练的环境。

| 组件 | 版本 |
|---|---|
| 操作系统 | Windows 11 |
| 包管理器 | Miniconda3 26.7.1（安装在 `D:\Miniconda3`） |
| conda 环境 | `comsen`，Python **3.12.14** |
| PyTorch / torchvision | **2.11.0+cu128** / 0.26.0+cu128 |
| OpenCV | opencv-contrib-python 5.0.0.93（需 contrib 版才有 CSRT / KCF 等经典 Tracker） |
| NumPy / Pandas | 2.5.3 / 3.0.6 |
| Matplotlib / Pillow | 3.11.2 / 12.3.0 |
| Ultralytics | 8.4.172 |
| ONNX / ONNX Runtime | onnx 1.23.1 / onnxruntime-gpu 1.22.0 |

**硬件**：NVIDIA GeForce RTX 5060 Laptop，8 GB 显存，算力架构 **sm_120**（Blackwell），驱动 616.56。

**两个关键决策**

1. **用 Python 3.12 而不是 3.14**：Miniconda base 自带 Python 3.14.7，但 PyTorch 尚未为 3.14 提供预编译轮子，因此单独创建了 Python 3.12 的 `comsen` 环境。
2. **装 cu128 版 PyTorch 而不是系统 CUDA toolkit**：RTX 5060 是 Blackwell（sm_120），本机原有的 CUDA 11.8 不支持该架构；而 `cu128` 版 wheel 已自带 CUDA 运行时（cuBLAS、cuDNN 等），只要驱动版本满足即可。这样省掉了约 3 GB 的完整 toolkit 安装，也不影响系统其他组件。

另外 `pip` 已永久配置清华镜像源，加速国内下载。

**验证记录**

```
torch.__version__                    = 2.11.0+cu128
torch.cuda.is_available()            = True
torch.cuda.get_device_name(0)        = NVIDIA GeForce RTX 5060 Laptop GPU
torch.cuda.get_device_capability(0)  = (12, 0)          # 即 sm_120
```

GPU 上 1000×1000 矩阵乘法实跑通过，说明 CUDA kernel 可正常编译执行。

该环境随后在后续三个任务中稳定使用——Task 2（OpenCV 图像处理与视频追踪）、Task 3（MLP 训练 12 epoch 共 14 秒）、Task 4（YOLO 训练 60 epoch 约 25 分钟）——这本身就是环境可用的证明。

> Task 1 第 5 步要求学习 Python 基础语法（参考菜鸟教程），属自学部分，无代码产出。

### Task 2 · OpenCV 与 Pillow

| 题目 | 内容 | 关键结论 |
|---|---|---|
| 题一 | 读取图片并分离 BGR 三通道 | 掌握了 OpenCV 的 BGR 通道顺序陷阱 |
| 题二 | 缩放 / 旋转 / 平移 / 翻转 | 理解了仿射变换矩阵与插值方式 |
| 题三 | 轮廓检测与面积周长计算 | 用 `THRESH_BINARY_INV + OTSU` 正确提取飞机轮廓 |
| 题四 | MOG2 背景减除 vs 帧差法 | MOG2 前景占比 2.25% / 检出率 99.5%，优于帧差法的 1.27% / 95.0% |
| 题五 | Tracker 追踪小球并分析准确度 | example3 上 KCF 最优（sim 0.954）；example4 因遮挡严重全部失效，分析了原因 |

详见随邮件提交的 Task 2 报告（`task2_opencv与pillow/报告.*`）

### Task 3 · 神经网络与 MLP

| 题目 | 内容 | 关键结论 |
|---|---|---|
| 题一 | 深度学习概念理解 | 神经元、激活函数、反向传播、优化器、过拟合等 |
| 题二 | 梯度下降拟合二元函数 | 7 个参数全部反解成功，对真值 R² = **0.999878** |
| 题三 | MLP 识别 MNIST | 784→256→128→10，测试准确率 **97.83%**，训练仅需 14 秒 |

详见随邮件提交的 Task 3 报告（`task3_MLP识别MNIST/报告.*`）

### Task 4 · YOLO 目标检测与模型量化（提高部分已完成）

| 步骤 | 内容 | 关键结论 |
|---|---|---|
| 第 2 步 | 预训练权重推理三张样例图 | bear 0.963 / bus 0.940 / person 0.919 |
| 第 3 步 | 无人机数据集训练 | drone 类 **mAP50 = 0.946**，召回 0.933 |
| 第 4 步 | 视频推理与指标分析 | 有效帧检出率 **100%**（低阈值）/ 87.7%（默认阈值） |
| 第 5 步 | ONNX + fp16/int8 量化 | 体积压缩至 **28.9%**，精度损失仅 2.6% |

**一个额外的深度发现**（详见报告第五章）：

1. **int8 量化在本机 GPU 上反而变慢**（86.5 fps vs fp32 ONNX 的 167.5 fps）——量化压缩体积是确定的，但速度收益强依赖硬件与推理后端

详见随邮件提交的 Task 4 报告（`task4_YOLO无人机检测/报告.*`）

---

## 运行环境

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Laptop（8 GB，sm_120） |
| Python | 3.12.14（conda 环境 `comsen`） |
| PyTorch | 2.11.0+cu128 |
| OpenCV | opencv-contrib-python 5.0.0.93 |
| Ultralytics | 8.4.172 |
| ONNX Runtime | onnxruntime-gpu 1.22.0（需 CUDA 12 版本） |

---

## 复现方式

各任务代码均为可直接运行的独立脚本，脚本内路径为绝对路径，运行前请按需调整。

```bash
# Task 2
python task2_opencv与pillow/src/run_all.py

# Task 3
python task3_MLP识别MNIST/src/q2_fit.py     # 拟合实验
python task3_MLP识别MNIST/src/q3_mlp.py     # MLP 识别 MNIST

# Task 4（按顺序）
python task4_YOLO无人机检测/src/step2_img_inference.py
python task4_YOLO无人机检测/src/step3_train_drone.py
python task4_YOLO无人机检测/src/step4_video_inference.py
python task4_YOLO无人机检测/src/step5_quantize.py
```

---

## 提交材料清单

按题目要求，本次提交包含：

1. **项目源码** → 本仓库（只放代码）
2. **实验报告 / 结果图片 / 结果视频 / 模型权重** → 已按 task 分类打包，发送至 `3813579841@qq.com`
3. **提交说明** → 题目仓库的 Issue 下有提交信息
