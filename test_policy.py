import torch
from policy import BCPolicy

bcpolicy = BCPolicy(12, 3)




t = torch.randn(32, 12)
act = bcpolicy(t)

print("Input shape:", t.shape)
print("Output shape:", act.shape)

assert t.shape == (32, 12)
assert act.shape == (32, 3)

print("MLP 测试通过！")