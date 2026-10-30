# BC-001 知识点总结：从 SAC 专家示范到 Behavior Cloning

> 项目：`panda_bc`\
> 环境：Windows + Conda（当前使用的环境名：`robot-sac-her`）\
> 任务：`PandaReach-v3`\
> 项目定位：P01（SAC+HER）之后、P02（MuJoCo）之前的短期过渡项目\
> 文档说明：本文记录当前对话中已经完成或明确讨论的内容。没有实际核验的事项会标注为"待确认"，不把推测写成实验事实。

------------------------------------------------------------------------

## 1. 项目目标与整体流程

### 1.1 项目要回答的问题

使用 P01 训练得到的 SAC+HER 策略作为
Teacher，采集成功的专家示范，再训练一个 Behavior Cloning（BC）策略作为
Student。

核心问题：

-   如何把机器人与环境交互产生的专家轨迹整理成监督学习数据？
-   如何用 PyTorch 的 Dataset、DataLoader 和 MLP 训练状态到动作的映射？
-   训练集 Loss、验证集 Loss 与真实环境成功率分别说明什么？
-   BC 能在多大程度上复现 SAC+HER 专家的任务表现？

### 1.2 项目数据流

``` text
SAC+HER Teacher
      |
      | env.reset() / model.predict() / env.step()
      v
成功 Episode（每条 Episode 是多个 (observation, action)）
      |
      | 展平并保存，同时记录 episode_lengths
      v
data/expert_demos.npz
      |
      | 根据 episode_lengths 恢复 Episode 边界
      | 按 Episode 划分 train / validation
      v
train_obs + train_actions       val_obs + val_actions
      |                               |
      v                               v
BCDataset                         BCDataset
      |                               |
      v                               v
DataLoader                       DataLoader
      |                               |
      v                               v
BCPolicy MLP 预测动作             BCPolicy 预测动作
      |                               |
      v                               v
MSE Loss + Backprop + Adam       Validation Loss（不更新参数）
      |
      v
保存 state_dict
      |
      v
加载模型，在 PandaReach-v3 中执行
      |
      v
统计 100 个固定种子的成功率
```

------------------------------------------------------------------------

## 2. 专家数据采集

### 2.1 一个 Transition 是什么

本项目把一个监督学习样本定义为：

\[ (o_t, a_t) \]

其中：

-   (o_t)：执行动作之前，机器人和目标的当前状态；
-   (a_t)：SAC Teacher 在这个状态下给出的动作。

采集时先用当前 `obs` 组成 Observation 向量，再由 Teacher
预测动作，然后把 `(obs_vector, action.copy())` 保存起来，最后执行
`env.step(action)`。

因此保存的是**当前状态与当前动作的配对**，不是 `(next_obs, action)`。

### 2.2 Observation 的组成

PandaReach-v3 返回的 Observation 是字典。本项目按固定顺序拼接：

``` python
obs_vector = np.concatenate([
    obs["observation"],
    obs["achieved_goal"],
    obs["desired_goal"]
]).astype(np.float32)
```

维度：

  字段                         Shape 含义
  ------------------------ --------- --------------------
  `obs["observation"]`        `(6,)` 机器人当前状态
  `obs["achieved_goal"]`      `(3,)` 当前实际达到的位置
  `obs["desired_goal"]`       `(3,)` 期望目标位置
  拼接结果                   `(12,)` BC 网络输入

Action 的单样本 Shape 为 `(3,)`。

所以一个样本是：

``` text
Observation: (12,)
Action:      (3,)
```

网络学习的映射是：

\[ `\pi`{=tex}\_`\theta`{=tex}(o_t)`\rightarrow `{=tex}`\hat `{=tex}a_t
\]

注意：网络输入是 Observation，专家 Action 是监督标签。**不是把
Observation 和 Action 一起作为网络输入。**

### 2.3 Episode 与 Transition

-   Transition：一组 `(observation, action)`。
-   Episode：一条完整轨迹，由多个 Transition 组成。
-   `expert_episodes`：多条 Episode 组成的外层列表。
-   `episode_lengths`：每条 Episode 包含多少个 Transition。

采集代码在 Episode 结束后检查 `info["is_success"]`，只保留成功
Episode。当前采集版本的逻辑是：每次调用 `collect_one_episode()`
创建一个新的 `transitions` 列表；成功才把整条 Episode 放入
`expert_episodes`。

### 2.4 当前专家数据统计（最新覆盖版本）

  指标                             结果
  ----------------------- -------------
  总尝试次数                        100
  成功 Episode 数量                 100
  Transition 总数                   288
  平均 Episode 长度                2.88
  最小 Episode 长度                   1
  最大 Episode 长度                   7
  Episode 长度中位数                  3
  Episode 长度标准差               0.93
  Observation 总 Shape      `(288, 12)`
  Action 总 Shape            `(288, 3)`
  Episode lengths Shape        `(100,)`

数据完整性检查：

``` python
assert sum(episode_lengths) == len(observations) == len(actions)
```

含义：所有 Episode 的长度之和必须等于 Observation 和 Action 的样本数。

**重要记录：** 最初采集版本有 269 个 Transition，后来重新采集并覆盖了
`data/expert_demos.npz`，当前文件是 288 个
Transition。旧模型与新数据不能混为同一实验版本。

### 2.5 当前尚未完成的专家数据核验

虽然已经检查了 Shape、Episode
数量、长度统计和长度总和，但以下检查没有被完整记录为已完成：

-   所有 Observation / Action 是否都没有 NaN 或 Inf；
-   Action 是否全部处于环境动作范围；
-   是否逐条复核所有 Episode 的状态---动作配对；
-   `info["is_success"]` 在环境中的具体实现语义是否经过源码级核验。

当前项目评估统一采用 Episode 最后一步的 `info["is_success"]`
作为成功判定。

------------------------------------------------------------------------

## 3. NPZ、Episode 划分与数据流

### 3.1 为什么不直接把 NPZ 交给 DataLoader？

NPZ 保存的是压缩的 NumPy 数组文件。当前文件中主要有：

-   `observations`：所有 Episode 展平后的 Observation；
-   `actions`：与 Observation 逐行对应的 Action；
-   `episode_lengths`：每条 Episode 的长度。

NPZ 是**存储格式**，不是 PyTorch 的 Dataset 接口。

我们需要先：

1.  加载 NPZ；
2.  根据 `episode_lengths` 恢复每条 Episode 的边界；
3.  按完整 Episode 划分训练集和验证集；
4.  把选中的 Episode 重新拼接成训练数组和验证数组；
5.  用 Dataset 封装成 PyTorch 可按索引读取的样本；
6.  用 DataLoader 组成 Batch。

并不是说 NPZ 不能被直接读取，而是本项目需要先按 Episode
划分，防止同一条轨迹的部分 Transition 同时进入训练集和验证集。

### 3.2 恢复 Episode 的起止索引

``` python
episode_ends = np.cumsum(episode_lengths)
episode_starts = np.concatenate(([0], episode_ends[:-1]))
```

例如：

``` text
episode_lengths = [2, 3, 1]

episode_starts = [0, 2, 5]
episode_ends   = [2, 5, 6]
```

对应切片：

``` text
Episode 0: observations[0:2]
Episode 1: observations[2:5]
Episode 2: observations[5:6]
```

Python 切片的右边界不包含在结果中。

### 3.3 为什么按 Episode 划分？

如果先把全部 Transition 随机打乱，再拆训练集和验证集，同一条 Episode
的相邻状态可能同时出现在两边。

这会让验证集与训练集过于相似，验证结果可能偏乐观。

本项目采用：

``` python
rng = np.random.default_rng(42)
episode_ids = rng.permutation(len(episode_lengths))

train_ids = episode_ids[:80]
val_ids = episode_ids[80:]
```

即 100 条 Episode 中，80 条用于训练，20 条用于验证。随机种子 42
用于保证划分可复现。

### 3.4 为什么需要 list，再 concatenate？

提取每条 Episode 时：

``` python
obs_list.append(observations[start:end])
action_list.append(actions[start:end])
```

此时列表中每个元素是一条 Episode 的二维数组。Observation 和 Action
使用相同的 Episode 编号、相同的 `start:end`，所以仍然逐行对应。

最后：

``` python
train_obs = np.concatenate(obs_list, axis=0)
train_actions = np.concatenate(action_list, axis=0)
```

`axis=0` 表示沿第 0 维（样本行方向）拼接。

拼接后得到连续的二维训练数组。Observation 与 Action 仍然逐行配对：

``` text
train_obs[i]  <-->  train_actions[i]
```

不能把 Observation 和 Action 分别用不同的 Episode
顺序或不同的切片拼接，否则会破坏监督标签对应关系。

### 3.5 常见 NumPy 操作

  操作                          本项目中的作用
  ----------------------------- ---------------------------------------
  `np.load()`                   加载 NPZ 文件
  `np.cumsum()`                 根据 Episode 长度累计计算边界
  `np.concatenate()`            拼接 Episode，或拼接 Observation 字段
  `np.stack()`                  把同形状的样本堆叠成批量数组
  `np.asarray()`                将列表转换为 NumPy 数组
  `np.random.default_rng(42)`   创建可复现的随机数生成器
  `np.intersect1d()`            检查训练与验证 Episode 编号是否重叠

------------------------------------------------------------------------

## 4. Dataset：把一条样本定义成可索引的数据

项目中的 `BCDataset` 继承 PyTorch 的 `Dataset`。

核心结构：

``` python
class BCDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.tensor(
            observations, dtype=torch.float32
        )
        self.actions = torch.tensor(
            actions, dtype=torch.float32
        )

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        observation = self.observations[idx]
        action = self.actions[idx]
        return observation, action
```

### 4.1 三个核心方法

-   `__init__()`：接收 NumPy 数组并转换成 `float32` Tensor。
-   `__len__()`：返回样本数量。
-   `__getitem__(idx)`：返回第 `idx` 个 Observation 和对应 Action。

例如：

``` python
observation, action = train_dataset[5]
```

本质上是实例通过索引语法调用 `__getitem__(5)`。

单样本 Shape：

``` text
observation: torch.Size([12])
action:      torch.Size([3])
```

### 4.2 Dataset 带来的价值

Dataset
不会凭空增加数据，也不会改变样本的语义。它主要是把"如何根据索引取出一条样本"封装起来，让
DataLoader 可以统一读取。

------------------------------------------------------------------------

## 5. DataLoader：把单样本组织成 Batch

项目使用：

``` python
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
```

### 5.1 Batch 维度

一个样本：

``` text
Observation: (12,)
Action:      (3,)
```

32 个样本组成一个 Batch：

``` text
Observation Batch: (32, 12)
Action Batch:      (32, 3)
```

第一维是 Batch 中的样本数量，第二维是每个样本的特征维度。

### 5.2 关键参数

-   `batch_size=32`：每个 Batch 最多包含 32 条样本。
-   `shuffle=True`：每轮训练时打乱训练样本顺序。
-   `drop_last=False`：最后一个不满 32 条的 Batch 也保留。
-   验证集使用 `shuffle=False`：让验证顺序稳定，便于复查。

例如，旧版 219 条训练样本在 `batch_size=32`、`drop_last=False`
时，会得到 6 个完整 Batch 和 1 个 27 条样本的 Batch，共 7 个 Batch。

当前 288 Transition
数据集的训练集和验证集样本数，应以当前划分脚本实际打印结果为准，不要直接沿用旧版的
219/50。

------------------------------------------------------------------------

## 6. MLP 策略网络

项目的 `BCPolicy` 使用 PyTorch `nn.Module` 定义 MLP：

``` python
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.mlp_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.mlp_net(obs)
```

### 6.1 网络结构

``` text
输入 Observation: 12 维
       |
       v
Linear(12, 64)
       |
      ReLU
       |
       v
Linear(64, 64)
       |
      ReLU
       |
       v
Linear(64, 3)
       |
       v
输出预测 Action: 3 维
```

网络学习：

\[ `\hat `{=tex}a_t = `\pi`{=tex}\_`\theta`{=tex}(o_t) \]

输入为 12 维状态向量，输出为 3 维动作向量。

### 6.2 为什么输出层没有 ReLU？

当前输出层是普通 `Linear`，没有使用 ReLU
限制动作正负方向。模型可以输出连续的正值或负值。

在环境执行阶段，代码使用：

``` python
action = np.clip(
    action,
    env.action_space.low,
    env.action_space.high
).astype(np.float32)
```

把预测动作裁剪到环境允许的动作范围内。

这是一项执行阶段的安全边界处理，不等同于网络内部已经保证输出范围合法。

------------------------------------------------------------------------

## 7. BC 的监督学习目标：MSE Loss

BC 使用专家示范中的 Action 作为监督标签。

对于一个 Batch：

-   `obs_batch`：网络输入；
-   `act_batch`：专家动作标签；
-   `act_output`：网络预测动作。

``` python
act_output = policy(obs_batch)
loss = criterion(act_output, act_batch)
```

损失函数：

\[ L(`\theta`{=tex})=`\frac{1}{N}`{=tex}`\sum`{=tex}*{i=1}\^{N}
`\left`{=tex}\|`\pi`{=tex}*`\theta`{=tex}(o_i)-a_i\^{expert}`\right`{=tex}\|\_2\^2
\]

PyTorch：

``` python
criterion = nn.MSELoss()
```

默认 `MSELoss` 会对预测张量和目标张量中所有元素计算均方误差并取平均。

本项目的目标是让网络预测动作尽可能接近专家动作。

**注意：MSE 低代表动作预测误差较小，不直接等于机器人任务成功率高。**

------------------------------------------------------------------------

## 8. 反向传播、梯度与优化器

项目使用 Adam：

``` python
optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=1e-3
)
```

一个 Batch 的训练更新顺序：

``` python
optimizer.zero_grad()

act_output = policy(obs_batch)
loss = criterion(act_output, act_batch)

loss.backward()

optimizer.step()
```

### 8.1 每一步的作用

1.  `zero_grad()`：清空上一轮累积的梯度。
2.  `policy(obs_batch)`：前向传播，计算预测动作。
3.  `criterion(...)`：计算预测动作与专家动作的 Loss。
4.  `loss.backward()`：反向传播，计算各个可训练参数的梯度。
5.  `optimizer.step()`：优化器根据梯度更新网络参数。

### 8.2 梯度的理解

梯度表示 Loss 对参数的局部变化率。优化器利用梯度信息决定参数如何更新。

更准确地说：

-   梯度不是简单等于"参数更新方向"；
-   对普通梯度下降而言，负梯度方向是局部使 Loss 下降的方向；
-   Adam 会结合梯度的一阶、二阶动量估计调整更新幅度。

### 8.3 为什么要检查参数是否改变？

单次训练更新时曾检查：

-   Loss 是否为标量；
-   梯度是否存在；
-   梯度是否为有限数值；
-   参数更新前后是否变化。

这可以验证训练更新链路确实执行了，而不只是前向传播得到一个输出。

------------------------------------------------------------------------

## 9. Epoch、训练 Loss 与验证 Loss

### 9.1 Epoch 是什么？

一个 Epoch 表示训练集中的所有 Batch 都被遍历一次。

本项目使用 50 Epoch 训练。每个 Epoch 中：

1.  遍历 `train_loader`；
2.  每个 Batch 执行前向传播、计算 Loss、反向传播和参数更新；
3.  统计该 Epoch 的训练平均 Loss；
4.  遍历 `val_loader` 计算验证 Loss。

### 9.2 为什么验证集每个 Epoch 都会计算？

验证集用于观察当前模型对未参与参数更新的数据的预测表现。

每个 Epoch
结束后，模型参数已经经过一轮训练，因此可以用新的模型参数重新计算验证
Loss。

验证过程：

``` python
policy.eval()

with torch.no_grad():
    for obs_batch, act_batch in val_loader:
        act_output = policy(obs_batch)
        loss = criterion(act_output, act_batch)
```

验证阶段：

-   不调用 `loss.backward()`；
-   不调用 `optimizer.step()`；
-   不更新模型参数；
-   `torch.no_grad()` 关闭梯度记录，减少计算和内存开销。

`policy.eval()` 与 `torch.no_grad()`
作用不同：前者切换模块的评估行为，后者关闭梯度记录。

### 9.3 为什么按样本数计算平均 Loss？

最后一个 Batch 可能不满 32 条。如果直接对各 Batch 的 Loss
做简单平均，每个 Batch 的权重相同，最后一个小 Batch 会被过度赋权。

因此采用：

``` python
sum_loss += loss.item() * obs_batch.shape[0]
sum_num += obs_batch.shape[0]

epoch_loss = sum_loss / sum_num
```

这是按样本数加权的 Batch Loss 平均方式。

### 9.4 如何理解训练与验证 Loss？

-   Train Loss 下降：模型越来越贴近训练专家样本。
-   Val Loss
    下降：模型在未参与参数更新的验证样本上也越来越贴近专家动作。
-   Train Loss 下降但 Val Loss 上升：需要警惕过拟合。
-   两者都下降：说明当前训练过程中拟合效果在改善，但仍需真实环境评估。

**验证 Loss 不是机器人成功率。** 最终是否完成 Reach
任务，必须把策略放回环境中执行。

------------------------------------------------------------------------

## 10. 模型保存与加载

训练结束后保存参数：

``` python
torch.save(policy.state_dict(), "bc_policy.pt")
```

加载：

``` python
policy = BCPolicy(obs_dim=12, action_dim=3)
state_dict = torch.load("bc_policy.pt", map_location=device)
policy.load_state_dict(state_dict)
policy.eval()
```

`state_dict` 保存网络参数，不包含完整的 Python
类定义。因此加载前仍然需要实例化相同结构的 `BCPolicy`。

建议保留模型与数据版本对应关系，例如：

``` text
bc_policy_269.pt  # 269 Transition 数据集训练的旧模型（若文件仍保留）
bc_policy_288.pt  # 288 Transition 数据集训练的新模型（若已按此名称保存）
```

实际文件名以项目目录为准，不要仅根据示例假设文件存在。

------------------------------------------------------------------------

## 11. 在 PandaReach-v3 中评估策略

### 11.1 BC 推理流程

每个 Episode：

1.  `env.reset(seed=eval_seed)`；
2.  从环境 Observation 字典中取出三个字段；
3.  拼接为 12 维 NumPy 向量；
4.  转为 `float32` Tensor；
5.  输入 BCPolicy 得到 3 维预测动作；
6.  转回 NumPy；
7.  用 `np.clip()` 裁剪到环境动作范围；
8.  `env.step(action)`；
9.  直到 `terminated or truncated`；
10. 读取最后一步的 `info["is_success"]`。

推理阶段使用 `torch.no_grad()`，因为不需要计算梯度。

### 11.2 为什么不需要 Batch 维度？

训练时输入通常是：

``` text
(32, 12)
```

评估时一次只输入一个状态：

``` text
(12,)
```

PyTorch `Linear` 可以处理单个一维输入，也可以处理带 Batch
维度的二维输入。因此这里不必为了推理强行扩展成 32 个重复样本。

### 11.3 固定种子评估

BC 与 SAC+HER 都使用：

``` python
eval_seeds = range(1000, 1100)

for eval_seed in eval_seeds:
    obs, info = env.reset(seed=eval_seed)
```

即 100 个种子，每个种子运行一个 Episode；不是 100 个种子各运行 100 个
Episode。

两种策略使用同一组种子，有助于控制初始条件差异。

------------------------------------------------------------------------

## 12. 实验结果与结论

### 12.1 已记录的实验结果

  实验                     专家数据版本 评估方式                成功率
  ----------------- ------------------- --------------------- --------
  BC 第一次训练          269 Transition 固定种子 1000～1099        95%
  BC 第二次训练          288 Transition 固定种子 1000～1099        83%
  SAC+HER Teacher     原始 SAC+HER 模型 固定种子 1000～1099       100%

SAC+HER 评估还记录：

-   Mean Reward：`-1.8700`
-   Successful Episodes：`100/100`

### 12.2 环境一致性核查

已经查看 `make_env()`：

``` python
def make_env(env_id, seed=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)

    return env
```

它只创建环境，并在提供 seed 时调用一次 `reset()`，没有添加额外
Wrapper，也没有改变 Observation 或 Action。

BC 使用 `gym.make("PandaReach-v3", render_mode="human")`；SAC 评估使用
`make_env(..., render_mode="rgb_array")`。两者渲染模式不同，但任务环境
ID 相同。

### 12.3 Reward 配置纠正

项目曾使用过注释：

``` text
PandaReach SAC Dense Baseline
```

但实际环境 ID 是 `PandaReach-v3`，且没有显式设置
`reward_type="dense"`。因此不能仅凭注释认定实验使用 Dense Reward。

当前记录应写为：

``` text
PandaReach-v3（默认奖励配置；本项目未显式指定 Dense Reward）
```

后续如果要确认奖励实现细节，应查看当前安装版本的 Panda-Gym
环境定义，而不是依据 YAML 注释推断。

### 12.4 目前可以得出的结论

-   BC 能够从 SAC+HER 专家示范中学习有效的状态---动作映射。
-   在当前 PandaReach-v3 任务和评估条件下，288 Transition 版本的 BC
    成功率为 83%。
-   SAC+HER Teacher 在相同固定种子集合上的成功率为 100%。
-   BC 的监督 Loss 与环境任务成功率不是同一个指标。
-   269 Transition 版本 BC 的 95% 与 288 Transition 版本的 83%
    说明：样本数量增加不保证单次训练的成功率一定提高。

### 12.5 不能过度得出的结论

当前实验还不足以证明：

-   BC 一般只能达到 SAC 的 83%；
-   269 个样本一定比 288 个样本更好；
-   BC 已经在复杂机器人操作任务中具备同样表现；
-   当前两次训练结果的差异完全由数据集大小导致。

数据采集结果、训练初始化、训练随机性和示范状态分布都可能影响结果。若要研究这些因素，需要额外控制变量和重复实验；当前
BC-001 不再扩大实验范围。

------------------------------------------------------------------------

## 13. 当前项目结构建议

以下是建议的组织方式，实际文件名以本地目录为准：

``` text
panda_bc/
├── data/
│   └── expert_demos.npz
├── dataset.py
├── policy.py
├── prepare_dataset.py
├── train_bc.py
├── evaluate_bc.py
├── test_train_step.py
├── README.md
├── 知识点.md
└── bc_policy.pt
```

可能还存在专家数据采集脚本、配置文件或其他测试脚本；提交前应以本地实际目录为准。

------------------------------------------------------------------------

## 14. GitHub 提交前检查清单

-   [ ] README 中不包含个人电脑绝对路径。
-   [ ] 检查 `expert_demos.npz` 和模型文件大小。
-   [ ] 如果希望别人复现实验，保留数据集或提供清楚的数据获取说明。
-   [ ] 不提交 Conda 环境目录、缓存、临时日志和无关视频。
-   [ ] 检查代码中是否存在个人路径、密钥或私人信息。
-   [ ] 检查 README 中的命令是否与真实文件名一致。
-   [ ] 记录当前实验数据版本为 288 Transition。
-   [ ] 记录旧版 269 Transition 结果，避免与新版混淆。
-   [ ] 完成 Git commit，并确认 `git status` 干净。

------------------------------------------------------------------------

## 15. 关键概念速查

  概念                       一句话理解
  -------------------------- ----------------------------------------------------
  Transition                 一个状态与该状态下专家动作的配对
  Episode                    一条完整的交互轨迹
  `episode_lengths`          每条 Episode 的 Transition 数量
  `np.cumsum`                计算 Episode 的累计边界
  `np.concatenate(axis=0)`   沿样本维拼接多个数组
  Dataset                    定义如何按索引读取一条样本
  DataLoader                 把样本组织成 Batch 并提供迭代
  Batch                      一批样本，第一维是样本数
  MLP                        多层全连接神经网络
  Forward                    根据输入计算预测结果
  MSE Loss                   衡量预测动作与专家动作之间的均方误差
  `zero_grad`                清空上一次的梯度
  `backward`                 计算参数梯度
  `optimizer.step`           根据梯度更新参数
  Epoch                      完整遍历训练集一次
  Validation                 用未参与参数更新的数据检查泛化表现
  `eval()`                   切换模块到评估行为
  `no_grad()`                不记录梯度
  `state_dict`               网络参数字典
  Covariate Shift            学生策略执行后访问到的状态分布可能偏离专家数据分布

------------------------------------------------------------------------

## 16. BC-001 阶段总结

BC-001 完成了从专家数据采集、Episode 划分、PyTorch 数据组织、MLP
监督训练、模型保存，到真实环境策略评估的完整基础流程。

本项目的核心收获不是手写了多少行代码，而是理解了：

1.  数据的语义和维度必须从采集阶段一直保持正确；
2.  Observation 与 Action 必须逐行对应；
3.  Dataset 和 DataLoader 负责组织数据，不负责学习策略；
4.  MLP 负责状态到动作的映射；
5.  Loss 衡量动作拟合误差，环境成功率衡量任务执行效果；
6.  专家策略的能力不会因为训练了一个 BC 网络就自动完整转移给 Student。

**下一阶段：P02------MuJoCo 仿真、机器人状态与动作接口、学习策略接入。**
