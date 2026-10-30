
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BCDataset
from policy import BCPolicy


# ============================================================
# 1. 固定随机种子，便于复现实验
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. 读取专家数据，并按 Episode 划分训练集和验证集
# ============================================================

data = np.load("data/expert_demos.npz")

observations = data["observations"]
actions = data["actions"]
episode_lengths = data["episode_lengths"]

assert len(observations) == len(actions)
assert episode_lengths.sum() == len(observations)

# 计算每条 Episode 在连续数组中的起止位置
episode_ends = np.cumsum(episode_lengths)
episode_starts = np.concatenate(([0], episode_ends[:-1]))

# 打乱 Episode 编号，80 条用于训练，20 条用于验证
rng = np.random.default_rng(42)
episode_ids = rng.permutation(len(episode_lengths))

train_ids = episode_ids[:80]
val_ids = episode_ids[80:]


def extract_episodes(episode_ids):
    """提取指定 Episode 的 Observation 和对应 Action。"""
    obs_list = []
    action_list = []

    for episode_id in episode_ids:
        start = episode_starts[episode_id]
        end = episode_ends[episode_id]

        obs_list.append(observations[start:end])
        action_list.append(actions[start:end])

    return (
        np.concatenate(obs_list, axis=0),
        np.concatenate(action_list, axis=0)
    )


train_obs, train_actions = extract_episodes(train_ids)
val_obs, val_actions = extract_episodes(val_ids)


# ============================================================
# 3. 创建 Dataset 和 DataLoader
# ============================================================

train_dataset = BCDataset(train_obs, train_actions)
val_dataset = BCDataset(val_obs, val_actions)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False
)


# ============================================================
# 4. 初始化策略网络、损失函数和优化器
# ============================================================

policy = BCPolicy(obs_dim=12, action_dim=3)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=1e-3
)


# ============================================================
# 5. 多 Epoch 训练与验证
# ============================================================

num_epochs = 50

for epoch in range(num_epochs):

    # -------------------- 训练阶段 --------------------
    policy.train()

    train_loss_sum = 0.0
    train_num = 0

    for obs_batch, act_batch in train_loader:#每个eooch要跑完所有的batch

        # 清空上一批次累积的梯度
        optimizer.zero_grad()

        # 前向传播：预测动作
        pred_actions = policy(obs_batch)

        # 计算预测动作与专家动作之间的 MSE
        loss = criterion(pred_actions, act_batch)

        # 反向传播：计算梯度
        loss.backward()

        # 更新网络参数
        optimizer.step()

        # 按样本数量累计 Loss，正确处理最后一个 Batch
        batch_size = obs_batch.shape[0]
        train_loss_sum += loss.item() * batch_size
        train_num += batch_size

    train_loss = train_loss_sum / train_num

    # -------------------- 验证阶段 --------------------
    policy.eval()

    val_loss_sum = 0.0
    val_num = 0

    # 验证时不需要计算梯度，也不更新参数
    with torch.no_grad():

        for obs_batch, act_batch in val_loader:

            pred_actions = policy(obs_batch)
            loss = criterion(pred_actions, act_batch)

            batch_size = obs_batch.shape[0]
            val_loss_sum += loss.item() * batch_size
            val_num += batch_size

    val_loss = val_loss_sum / val_num

    # 每个 Epoch 只打印一行结果
    print(
        f"Epoch [{epoch + 1:02d}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

# 所有 Epoch 训练结束后保存模型参数
torch.save(policy.state_dict(), "bc_policy_288.pt")
print("BC 模型已保存：bc_policy_288.pt")
print("BC 训练结束。")
