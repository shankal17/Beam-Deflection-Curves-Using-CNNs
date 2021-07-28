import torch.nn as nn

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        self.conv_1 = nn.Conv1d(1, 2, 3)
        self.conv_2 = nn.Conv1d(2, 4, 3)
        self.conv_3 = nn.Conv1d(4, 4, 3)
        self.conv_4 = nn.Conv1d(4, 2, 3)
        self.conv_5 = nn.Conv1d(2, 1, 3)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(x)
        out = self.conv_3(x)
        out = self.conv_4(x)
        out = self.conv_5(x)

        return out



class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    n = ConvolutionalNetwork()
    print(n)