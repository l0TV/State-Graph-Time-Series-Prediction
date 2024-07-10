import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.tensor(weight, dtype=torch.float32).unsqueeze(0)

    def forward(self, predict, label):
        if predict.device != 'cpu':
            self.weight = self.weight.cuda()
        return torch.mean(self.weight * torch.pow(predict - label, 2))


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b = torch.tensor([1, 2], dtype=torch.float32).unsqueeze(-1)
    print(a * b)
    print(torch.mean(a, 0))
    print(torch.tensor([1, 2, 3]).unsqueeze(-1))
    print(a.device)
