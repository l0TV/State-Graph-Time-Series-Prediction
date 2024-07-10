import torch
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self, input_length, out_length, num_hidden_features=64, num_layers=5):
        super().__init__()
        self.input_length = input_length
        self.num_hidden_features = num_hidden_features
        self.num_layers = num_layers
        fc = [nn.Linear(input_length, num_hidden_features)]
        for i in range(self.num_layers - 1):
            fc.append(nn.Linear(input_length + num_hidden_features * (i + 1), num_hidden_features))
        self.fc1 = nn.ModuleList(fc)
        fc = []
        for i in range(self.num_layers):
            fc.append(nn.Linear(num_hidden_features * (i + 1), num_hidden_features))
        self.fc2 = nn.ModuleList(fc)
        self.bottleneck = nn.Linear(input_length + num_hidden_features * self.num_layers, num_hidden_features)
        # self.out_fc = nn.Linear(num_hidden_features * (self.num_layers + 1), out_length)
        self.out_fc = nn.Linear(input_length + num_hidden_features * self.num_layers, out_length)

    def forward(self, x):
        x_list = [x]
        for i in range(self.num_layers):
            x_in = torch.concat(x_list, dim=1)
            x_out = self.fc1[i](x_in)
            x_list.append(x_out)
        # x_list = [self.bottleneck(torch.concat(x_list, dim=1))]
        # for i in range(self.num_layers):
        #     x_in = torch.concat(x_list, dim=1)
        #     x_out = self.fc2[i](x_in)
        #     x_list.append(x_out)
        return self.out_fc(torch.concat(x_list, dim=1))


if __name__ == '__main__':
    net = DenseNet(20, 30)
    x = torch.rand([32, 20])
    y = net(x)
    print(y.shape)
