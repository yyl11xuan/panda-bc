# Panda BC --- Behavior Cloning for PandaReach

使用 SAC+HER 训练得到的专家策略采集示范数据，再通过 PyTorch Behavior
Cloning（BC）训练一个 MLP 策略，并在 `PandaReach-v3` 环境中评估。

本项目是机器人学习路线中的 BC-001 过渡项目，位于
P01（SAC+HER）之后、P02（MuJoCo）之前。

## 1. 项目目标

本项目完成以下流程：

1.  加载 SAC+HER Teacher；
2.  采集成功的专家 Episode；
3.  将 Observation、Action 和 Episode 长度保存为 NPZ；
4.  按 Episode 划分训练集和验证集；
5.  使用 PyTorch Dataset / DataLoader 组织数据；
6.  训练 MLP Behavior Cloning 策略；
7.  保存模型参数；
8.  在 `PandaReach-v3` 中评估成功率。

## 2. 技术栈

-   Python
-   NumPy
-   PyTorch
-   Gymnasium
-   Panda-Gym
-   PyBullet
-   Stable-Baselines3（专家模型加载、数据采集或 SAC 对照评估时使用）

项目在 Windows + Conda
环境中运行。实际依赖版本以运行环境为准；建议在复现前记录版本，不要直接假设不同机器上的版本完全相同。

## 3. 项目结构

``` text
panda_bc/
├── data/
│   └── expert_demos.npz
├── dataset.py
├── policy.py
├── prepare_dataset.py
├── train_bc.py
├── evaluate_bc.py
├── README.md
└── 知识点.md
```

实际目录可能包含专家数据采集脚本、测试脚本、模型文件或配置文件。请以仓库当前文件为准。

## 4. 专家数据格式

当前最终覆盖的数据集统计：

  字段                        Shape 含义
  ------------------- ------------- -----------------------------------
  `observations`        `(288, 12)` 288 个 Observation
  `actions`              `(288, 3)` 与 Observation 逐行对应的专家动作
  `episode_lengths`        `(100,)` 100 条 Episode 的长度

Observation 由三个字段按顺序拼接：

``` python
[
    obs["observation"],   # (6,)
    obs["achieved_goal"], # (3,)
    obs["desired_goal"]   # (3,)
]
```

因此单个 Observation 为 12 维，单个 Action 为 3 维。

最新数据统计：

-   尝试次数：100
-   成功 Episode：100
-   Transition：288
-   平均 Episode 长度：2.88
-   最短 / 最长 Episode：1 / 7
-   Episode 长度中位数：3
-   Episode 长度标准差：0.93

数据文件：`data/expert_demos.npz`。

## 5. 模型结构

MLP 输入 12 维 Observation，输出 3 维连续 Action：

``` text
12 → Linear(64) → ReLU
   → Linear(64) → ReLU
   → Linear(3)
```

训练目标是让预测动作接近专家动作，使用 `MSELoss`，优化器使用 Adam。

训练数据按 Episode 划分，当前代码采用 80% Episode 作为训练集、20%
Episode 作为验证集。DataLoader 的 Batch Size 为
32；训练集打乱，验证集不打乱，保留最后一个不完整 Batch。

## 6. 运行环境

先激活项目使用的 Conda 环境。例如：

``` powershell
conda activate robot-sac-her
```

需要安装并确保以下依赖可用：

``` text
numpy
torch
gymnasium
panda-gym
pybullet
stable-baselines3
```

如果运行 SAC 配置评估脚本，还需要：

``` text
PyYAML
```

注意：具体版本应以项目实际环境为准。建议在成功运行后导出并整理依赖版本，例如：

``` powershell
python --version
pip freeze
```

不要把整个 Conda 环境目录提交到 GitHub。

## 7. 运行流程

以下命令假设对应脚本存在于项目根目录，且数据文件已经准备好。

### 7.1 检查数据与划分

``` powershell
python prepare_dataset.py
```

### 7.2 训练 BC

``` powershell
python train_bc.py
```

训练脚本会遍历多个 Epoch，记录训练集和验证集
Loss。训练完成后保存模型参数。当前模型文件名以脚本中的 `torch.save(...)`
路径为准，例如 `bc_policy.pt`。

### 7.3 评估 BC

``` powershell
python evaluate_bc.py
```

评估脚本加载 BC 模型，在 `PandaReach-v3` 中执行策略，并统计成功率。

如需复现本项目的固定种子评估，BC 与 SAC+HER 都使用种子 `1000` 至
`1099`，每个种子运行一个 Episode。

### 7.4 SAC+HER 对照评估

SAC+HER Teacher 模型来自 P01
项目。其模型文件需要放在本地可访问的位置，并按 SAC
评估脚本中的路径加载。

原训练项目路径不属于本仓库的可移植路径。复现者需要自行提供 Teacher
模型，并修改本地配置或脚本中的模型路径。

## 8. 实验结果

固定测试种子：`1000–1099`，共 100 个 Episode。

  策略                   专家数据版本   成功率
  ------------------ ---------------- --------
  BC（第一次训练）     269 Transition      95%
  BC（第二次训练）     288 Transition      83%
  SAC+HER Teacher        P01 训练模型     100%

SAC+HER 评估记录：

-   Mean Reward：`-1.8700`
-   Successful Episodes：`100/100`

注意：

-   269 Transition 和 288 Transition 是两次不同的数据采集版本。
-   288 Transition 是当前 `expert_demos.npz` 的最新版本。
-   两次 BC
    训练的成功率不同，不能据此得出"数据量越大性能越差"的一般结论。
-   当前成功判定采用 Episode 最后一步的 `info["is_success"]`。
-   BC 与 SAC 使用相同的测试种子集合和相同的最终步成功判定口径。

## 9. 环境配置说明

当前使用 `PandaReach-v3`，BC 和 SAC 评估均创建同一个环境 ID。

SAC 评估辅助函数 `make_env()` 只创建环境，并可选地执行一次带种子的
`reset()`，没有添加额外 Wrapper。

BC 使用 `render_mode="human"`，SAC 评估使用
`render_mode="rgb_array"`。渲染模式不同，但没有在 `make_env()`
中发现会改变任务动力学或 Observation / Action 的额外处理。

曾经使用过 "Dense Baseline" 的配置注释，但实际环境 ID 是
`PandaReach-v3`，代码没有显式设置 `reward_type="dense"`。因此本项目
README 不将其标记为 Dense Reward 实验。

## 10. 复现注意事项

-   检查 Observation
    拼接顺序必须一致：`observation`、`achieved_goal`、`desired_goal`。
-   Observation 和 Action 必须逐行配对。
-   按 Episode 划分训练集和验证集，避免同一条轨迹跨集合泄漏。
-   评估时使用训练时相同的网络结构。
-   BC 推理时将单个 Observation 转为 `float32` Tensor。
-   环境执行动作前，将模型输出转换为 NumPy，并裁剪到 `env.action_space`
    范围。
-   使用 `policy.eval()` 和 `torch.no_grad()` 进行推理。
-   如果重新采集数据并覆盖 NPZ，要记录新数据版本，并重新训练
    BC；旧模型不会自动适配新数据。
-   记录模型文件对应的数据版本，避免把 269 Transition 和 288 Transition
    的模型混淆。

## 11. 文档

-   [`知识点.md`](知识点.md)：数据流、Episode 划分、Tensor
    维度、Dataset、DataLoader、MLP、Loss、反向传播、训练与评估的学习笔记。

## 12. 下一阶段

BC-001 完成后进入 P02：

-   MuJoCo 仿真环境；
-   MJCF 与机器人模型配置；
-   机器人状态读取和动作施加；
-   将学习策略接入仿真机器人。

本项目的目标是建立从专家示范数据到策略训练和环境评估的基础闭环，不代表已经完成
ACT、LeRobot 或 VLA 的系统学习。
