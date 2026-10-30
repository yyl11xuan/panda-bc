import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import panda_gym
import statistics


def collect_one_episode(env, model):
    obs, _ = env.reset()

    # ⭐ 修改1：每次调用函数，都创建一个新的 Episode 列表
    transitions = []

    done = False
    episode_length = 0

    while done == False:
        action, _ = model.predict(obs, deterministic=True)

        obs_vector = np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"]
        ]).astype(np.float32)

        episode_length += 1
        transitions.append((obs_vector, action.copy()))

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        obs = next_obs

    success = bool(info["is_success"])

    return transitions, success, episode_length


# 创建环境
env_1 = gym.make("PandaReach-v3")

# 加载 SAC Teacher
model_path = r"D:\robot-learning\robot-manipulation-sac-her\runs\PandaReach-v3_HER_seed42_steps50000\best_model.zip"
model = SAC.load(model_path, env=env_1, device="cuda")


# ⭐ 修改2：创建用于保存所有成功 Episode 的外层列表
expert_episodes = []
obs_list = []
action_list = []
episode_lengths = []
# ⭐ 修改3：分别设置成功目标和最大尝试次数
target_success = 100
max_attempts = 110

attempts = 0

# ⭐ 修改4：根据成功 Episode 数量控制循环
while len(expert_episodes) < target_success and attempts < max_attempts:

    transitions, success, episode_length = collect_one_episode(env_1, model)

    attempts += 1

    # ⭐ 修改5：成功才把整条 Episode 加入数据集
    if success:
        expert_episodes.append(transitions)

    print(f"第 {attempts} 次尝试：成功={success}，Episode长度={episode_length}")


# ⭐ 修改6：统计数据集，而不是只统计最后一次的 transitions
total_transitions = sum(len(ep) for ep in expert_episodes)

print("\n===== 数据采集结果 =====")
print(f"总尝试次数：{attempts}")
print(f"成功Episode数量：{len(expert_episodes)}")
print(f"总Transition数量：{total_transitions}")

if len(expert_episodes) > 0:
    avg_length = total_transitions / len(expert_episodes)
    print(f"平均Episode长度：{avg_length:.2f}")

    x, y = expert_episodes[0][0]
    print(f"Observation shape：{x.shape}")
    print(f"Action shape：{y.shape}")


for episode in expert_episodes:
    episode_lengths.append(len(episode))
    for obs, action in episode:
        obs_list.append(obs)
        action_list.append(action)



#上面都是在循环收集sac跑的数据，我们下面要把这些数据转换成NPZ格式，方便后续IL训练BC网络
observations = np.stack(obs_list)
actions = np.stack(action_list)
episode_lengths = np.asarray(episode_lengths, dtype=np.int32)
lens = episode_lengths
print("最大步长：",lens.max())
print("最小步长：",lens.min())
print("中位数：",np.median(lens))
print(f"mean  : {lens.mean():.2f}")
print(f"std   : {lens.std():.2f}")

assert sum(episode_lengths) == len(observations) == len(actions)

np.savez_compressed(
    "data/expert_demos.npz",
    observations=observations,
    actions=actions,
    episode_lengths=episode_lengths
)

data = np.load("data/expert_demos.npz")

print("Observations:", data["observations"].shape)# xxx,12
print("Actions:", data["actions"].shape)#xxx,3
print("Episode lengths:", data["episode_lengths"].shape)#100,

assert data["observations"].shape[1] == 12
assert data["actions"].shape[1] == 3
assert data["episode_lengths"].sum() == len(data["observations"])

env_1.close()

