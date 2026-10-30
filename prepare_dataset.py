import torch
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import BCDataset
from policy import BCPolicy


# 1. 读取专家示范数据
data = np.load("data/expert_demos.npz")

observations = data["observations"]
actions = data["actions"]
episode_lengths = data["episode_lengths"]

# 基础数据检查
assert len(observations) == len(actions)
assert episode_lengths.sum() == len(observations)
assert observations.shape[1] == 12
assert actions.shape[1] == 3


# 2. 计算每条 Episode 在数组中的起止位置
episode_ends = np.cumsum(episode_lengths)
episode_starts = np.concatenate(
    ([0], episode_ends[:-1])
)


# 3. 按 Episode 划分训练集和验证集
rng = np.random.default_rng(42)
episode_ids = rng.permutation(len(episode_lengths))#就是按长度生成了一个随机数组

train_ids = episode_ids[:80]
val_ids = episode_ids[80:]


# 4. 提取完整 Episode
train_obs_list = []
train_action_list = []

val_obs_list = []
val_action_list = []

for episode_id in train_ids:
    start = episode_starts[episode_id]
    end = episode_ends[episode_id]

    train_obs_list.append(observations[start:end])#这里最后的最后得到的是80个随机打乱的多个episode连续的obs,序号对应的是同样的action，这里的列表一个元素[i}算一个episode
    train_action_list.append(actions[start:end])

for episode_id in val_ids:
    start = episode_starts[episode_id]
    end = episode_ends[episode_id]

    val_obs_list.append(observations[start:end])
    val_action_list.append(actions[start:end])


# 5. 合并 Episode，形成二维数组
train_obs = np.concatenate(train_obs_list, axis=0)
train_actions = np.concatenate(train_action_list, axis=0)

val_obs = np.concatenate(val_obs_list, axis=0)
val_actions = np.concatenate(val_action_list, axis=0)


# 6. 创建 Dataset 实例
train_dataset = BCDataset(train_obs, train_actions)#放入类中大小形状无变化，仅仅类型变化了，且调用索引会变化
val_dataset = BCDataset(val_obs, val_actions)

# 7. 检查数据集长度
print("训练集长度:", len(train_dataset))
print("验证集长度:", len(val_dataset))

assert len(train_dataset) == len(train_obs)
assert len(val_dataset) == len(val_obs)


# 8. 检查单个样本
observation, action = train_dataset[5]

print("\n第 5 号样本")
print("Observation shape:", observation.shape)
print("Action shape:", action.shape)
print("Observation dtype:", observation.dtype)
print("Action dtype:", action.dtype)

assert observation.shape == (12,)
assert action.shape == (3,)
assert observation.dtype == torch.float32
assert action.dtype == torch.float32

print("\nDataset 检查通过！")



#9
train_dataLoader = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=False)#这里之后会得到（batch,dim)
val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=True,drop_last=False)

obs, act = next(iter(train_dataLoader))#取第一个batch
bcpolicy = BCPolicy(12,3)
optimizer = torch.optim.Adam(bcpolicy.parameters(), lr=1e-3)
optimizer.zero_grad()
act_output = bcpolicy(obs)
# 创建损失函数,优化器
criterion = nn.MSELoss()

old_w = bcpolicy.mlp_net[0].weight.detach().clone()
print("未更新前第一层参数：",old_w)
# 计算损失：预测动作与专家动作进行比较
loss = criterion(act_output, act)



loss.backward()

g = bcpolicy.mlp_net[0].weight.grad
print("梯度是：",g)
optimizer.step()

new_w = bcpolicy.mlp_net[0].weight
print("更新后第一层参数：",new_w)

