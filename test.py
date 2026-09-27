import torch

data = torch.load("best_model.pth")
print(type(data))
print(data.keys() if isinstance(data, dict) else "Pas un dict")
