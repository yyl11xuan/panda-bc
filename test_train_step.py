
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BCDataset
from policy import BCPolicy


# ============================================================
# 1. 读取专家数据，并按 Episode 划分训练集和验证集
# ============================================================

data = np.load("data/expert_demos.npz")

observations = data["observations"]
actions = data["actions"]
episode_lengths = data["episode_lengths"]

# 检查数据完整性
assert len(observations) == len(actions)
assert episode_lengths.sum() == len(observations)

# 计算每条 Episode 在连续数组中的起止位置
episode_ends = np.cumsum(episode_lengths)
episode_starts = np.concatenate(([0], episode_ends[:-1]))

# 固定随机种子，保证划分可以复现
rng = np.random.default_rng(42)
episode_ids = rng.permutation(len(episode_lengths))

train_ids = episode_ids[:80]
val_ids = episode_ids[80:]


def extract_episodes(episode_ids):
    """根据 Episode 编号提取状态和对应动作。"""
    obs_list = []
    action_list = []

    for episode_id in episode_ids:
        start = episode_starts[episode_id]
        end = episode_ends[episode_id]

        obs_list.append(observations[start:end])
        action_list.append(actions[start:end])

    # 将多条完整 Episode 沿第 0 维拼接
    return (
        np.concatenate(obs_list, axis=0),
        np.concatenate(action_list, axis=0)
    )


train_obs, train_actions = extract_episodes(train_ids)
val_obs, val_actions = extract_episodes(val_ids)


# ============================================================
# 2. 创建 Dataset 和 DataLoader
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

# 取出一个训练 Batch
obs, expert_actions = next(iter(train_loader))

# obs:            (32, 12)
# expert_actions: (32, 3)


# ============================================================
# 3. 创建 BC 策略网络、损失函数和优化器
# ============================================================

policy = BCPolicy(obs_dim=12, action_dim=3)

# MSE：计算预测动作与专家动作之间的均方误差
criterion = nn.MSELoss()

# Adam：根据参数梯度更新网络参数
optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=1e-3
)
sum_loss=0
sum_num = 0
sum_loss_val=0
sum_num_val = 0
for obs_batch, act_batch in train_loader:#这样循环就可以拿到datalodaer里面的东西
    optimizer.zero_grad()
    act_output = policy(obs_batch)
    Loss = criterion(act_batch,act_output)
    sum_loss += obs_batch.shape[0]*Loss.item()
    sum_num += obs_batch.shape[0]
    Loss.backward()

    optimizer.step()
print("样本总数：",sum_num)
print(f"总的平均loss = {sum_loss / sum_num}")


policy.eval()


with torch.no_grad():
    # 在这里执行验证
    for obs_batch, act_batch in val_loader:#这样循环就可以拿到datalodaer里面的东西
        act_output = policy(obs_batch)
        Loss = criterion(act_batch,act_output)
        sum_loss_val += obs_batch.shape[0]*Loss.item()
        sum_num_val += obs_batch.shape[0]

    print("验证集样本总数：",sum_num_val)
    print(f"验证集总的平均loss = {sum_loss_val / sum_num_val}")




# # ============================================================
# # 4. 保存更新前的参数，用于验证参数是否发生变化
# # ============================================================

# # 这里选择第一层 Linear 的权重作为观察对象
# old_w = policy.mlp_net[0].weight.detach().clone()


# # ============================================================
# # 5. 执行一次完整的训练更新
# # ============================================================

# # 第一步：清空之前累积的梯度
# optimizer.zero_grad()

# # 第二步：前向传播，预测动作
# predicted_actions = policy(obs)

# # 第三步：计算预测动作与专家动作之间的 Loss
# loss = criterion(predicted_actions, expert_actions)

# # 第四步：反向传播，计算网络参数的梯度
# loss.backward()

# # 检查第一层权重是否获得梯度
# grad = policy.mlp_net[0].weight.grad

# assert grad is not None
# assert torch.isfinite(grad).all().item()
# assert torch.isfinite(loss).item()

# # 第五步：优化器根据梯度更新网络参数
# optimizer.step()


# # ============================================================
# # 6. 验证本次更新是否成功
# # ============================================================

# new_w = policy.mlp_net[0].weight.detach().clone()

# # 判断更新前后的权重是否完全相同
# weights_changed = not torch.equal(old_w, new_w)

# # 计算权重变化的最大绝对值
# max_change = (new_w - old_w).abs().max().item()

# print("===== 单次训练更新结果 =====")
# print("Observation Shape:", obs.shape)
# print("专家 Action Shape:", expert_actions.shape)
# print("预测 Action Shape:", predicted_actions.shape)
# print("Loss Shape:", loss.shape)
# print("Loss Value:", loss.item())
# print("梯度存在:", grad is not None)
# print("梯度数值有限:", torch.isfinite(grad).all().item())
# print("参数发生变化:", weights_changed)
# print("参数最大变化量:", max_change)

# assert predicted_actions.shape == expert_actions.shape
# assert loss.ndim == 0
# assert weights_changed

# print("单次训练更新验收通过！")
