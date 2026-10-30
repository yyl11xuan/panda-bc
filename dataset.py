
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    def __init__(self, observations, actions):
        # 保存数据，并统一转换为 PyTorch float32 Tensor
        self.observations = torch.tensor(
            observations, dtype=torch.float32
        )
        self.actions = torch.tensor(
            actions, dtype=torch.float32
        )

        # 检查每个 Observation 是否都有对应的 Action
        if len(self.observations) != len(self.actions):
            raise ValueError(
                "Observation 和 Action 的样本数量不一致"
            )

    def __len__(self):
        # 返回样本数量，而不是每个样本的特征数
        return self.observations.shape[0]

    def __getitem__(self, idx):
        # 使用相同的索引，取出一对状态和专家动作
        observation = self.observations[idx]
        action = self.actions[idx]

        return observation, action
