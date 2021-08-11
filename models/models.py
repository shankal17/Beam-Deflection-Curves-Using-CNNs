import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder1d(nn.Module):
    def __init__(self):
        super(ConvAutoencoder1d, self).__init__()
        self.conv_1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv_2 = nn.Conv1d(16, 4, 3, padding=1)
        self.pooling_func = nn.MaxPool1d(2)

        self.trans_conv_1 = nn.ConvTranspose1d(4, 16, 2, stride=2)
        self.trans_conv_2 = nn.ConvTranspose1d(16, 1, 2, stride=2)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = self.pooling_func(out)
        out = F.relu(self.conv_2(out))
        out = self.pooling_func(out)

        out = F.relu(self.trans_conv_1(out))
        out = torch.sigmoid(self.trans_conv_2(out)) - 0.5
        return out


if __name__ == '__main__':
    m = ConvAutoencoder1d()
    input = torch.randn(1, 1, 100)
    print(input, input.shape)
    output = m(input)
    print(output)
    print(output.shape)