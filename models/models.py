import torch.nn as nn

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 2, 3, padding=1)
        self.conv_2 = nn.Conv1d(2, 4, 3, padding=1)
        self.conv_3 = nn.Conv1d(4, 16, 3, padding=1)
        self.conv_4 = nn.Conv1d(16, 4, 3, padding=1)
        self.conv_5 = nn.Conv1d(4, 2, 3, padding=1)
        self.conv_6 = nn.Conv1d(2, 1, 3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.leaky_relu(out)
        out = self.conv_2(out)
        out = self.leaky_relu(out)
        out = self.conv_3(out)
        out = self.leaky_relu(out)
        out = self.conv_4(out)
        out = self.leaky_relu(out)
        out = self.conv_5(out)
        out = self.leaky_relu(out)
        out = self.conv_6(out)

        return out


if __name__ == '__main__':
    import torch
    n = ConvolutionalNetwork()
    # print(n)
    input = torch.randn(1, 51).unsqueeze(1)
    print(input)
    print(n(input))