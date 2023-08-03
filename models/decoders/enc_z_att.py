import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, enc_size, z_size, attention_size):
        super(Attention, self).__init__()
        self.key = nn.Linear(enc_size, attention_size)
        self.value = nn.Linear(enc_size, attention_size)
        self.query = nn.Linear(z_size, attention_size)

    def forward(self, enc, z):
        # 计算键和值
        k = self.key(enc)
        v = self.value(enc)

        # 计算查询
        q = self.query(z)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)

        # 计算加权和
        weighted_sum = torch.matmul(attention_weights, v)

        return weighted_sum


# # 假设enc和z是两个已有的torch.Tensor张量
# enc = torch.randn(3, 164, 32)
# z = torch.randn(3, 1000, 5)
#
# # 初始化注意力机制模型
# attention = Attention(enc_size=32, z_size=5, attention_size=64)
#
# # 执行注意力操作
# output = attention(enc, z)
#
# # 打印输出张量的形状
# print(output.shape)