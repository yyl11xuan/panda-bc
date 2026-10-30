import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import panda_gym
from policy import BCPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("PandaReach-v3",render_mode="human")

policy = BCPolicy(12,3).to(device)

state_dict = torch.load("bc_policy_288.pt", map_location=device)
policy.load_state_dict(state_dict)

policy.eval()
env_episode = 100
success_episode = 0
# for _ in range(env_episode):
eval_seeds = list(range(1000, 1100))
for seed in eval_seeds:
    obs, _ = env.reset(seed=seed)
    #episode_success=False
    # obs_batch = obs_tensor.unsqueeze(0).expand(32, -1)#再扩充成张量（32，12），记住只有张量才能unsqueeze和expand
    done = False
    while done==False:
        obs_vector = np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"]
        ]).astype(np.float32)
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=device)#变成张量（12，）
        with torch.no_grad():
            action_tensor = policy(obs_tensor)#这里输出的当然也是张量
        action = action_tensor.squeeze(0).cpu().numpy()#这里是张量转numpy
        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high
        ).astype(np.float32)#这里把动作裁剪，让动作范围在环境内
        next_obs,reward,terminated,truncated,info = env.step(action)#环境需要numpy,模型需要张量

        done = terminated or truncated
        obs = next_obs
        #episode_success = episode_success or bool(info["is_success"])#只要中途到过目标点，就算做成功过
    if bool(info["is_success"]):#,这种写法是只查最后一步是否走到目标点成功
        success_episode+=1
    
    # if episode_success:
    #     success_episode+=1

print("成功率为：",success_episode/env_episode)

env.close()



