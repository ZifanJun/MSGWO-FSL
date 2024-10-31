import torch
import torch.nn as nn
import torch.nn.functional as Func

class Net(nn.Module):
    def __init__(self, input_channels=1):  # 默认设为1，适配MNIST和EMNIST
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 3 * 3, 10)  # 使用3x3的展平层以兼容28x28输入

    def forward(self, x):
        x = self.pool1(Func.relu(self.conv1(x)))
        x = self.pool2(Func.relu(self.conv2(x)))
        x = self.pool3(Func.relu(self.conv3(x)))

        # 动态计算展平尺寸
        flattened_size = x.size(1) * x.size(2) * x.size(3)  # 获取展平尺寸
        x = x.view(-1, flattened_size)  # 展平层
        x = self.fc1(x)
        return x

# class SelfAttentionForMask(nn.Module):
#     def __init__(self, input_size, hidden_size=128, num_heads=4):
#         super(SelfAttentionForMask, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
#         self.fc = nn.Linear(input_size, 1)
#
#     def forward(self, x):
#         attn_output, _ = self.attention(x, x, x)
#         out = self.fc(attn_output[:, -1, :])
#         return torch.sigmoid(out)