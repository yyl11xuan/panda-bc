import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_dim=64):
        super().__init__()
        # TODO：定义网络层
        self.mlp_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),#这里的输入实际是（32，12），过程就是变为（64，12）
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        # TODO：定义前向传播
        # act = self.mlp_net(obs)
        return self.mlp_net(obs)