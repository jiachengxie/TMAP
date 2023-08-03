import torch
import torch.nn as nn
import torch.nn.functional as F


class Traget_surround(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Traget_surround, self).__init__()
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.linear3 = nn.Linear(input_dim * 2, input_dim)

    def forward(self, target_features, surrounding_features):
        # target_features: 目标agent的特征表示 (batch_size, target_dim)
        # surrounding_features: 周围agent的特征表示 (batch_size, num_agents, surrounding_dim)
        target_expanded = target_features.unsqueeze(1).expand(-1, surrounding_features.size(1), -1)
        combined_features = torch.cat((target_expanded, surrounding_features), dim=2)
        attention_scores = F.leaky_relu(self.linear1(combined_features))
        attention_scores = self.linear2(attention_scores).squeeze(dim=2)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_features = torch.bmm(attention_weights.unsqueeze(1), surrounding_features).squeeze(dim=1)
        # 将目标agent的特征和注意力调整后的周围agent特征进行合并
        combined_output = torch.cat((target_features, attended_features), dim=1)
        combined_output = F.leaky_relu(self.linear3(combined_output))

        return combined_output

# # 创建一个示例输入
# batch_size = 32
# target_dim = 16
# num_agents = 10
# surrounding_dim = 16
#
# target_features = torch.randn(batch_size, target_dim)
# surrounding_features = torch.randn(batch_size, num_agents, surrounding_dim)
#
# # 创建PGPAttention模块实例
# attention = Traget_surround(target_dim, hidden_dim=64)
#
# # 使用PGPAttention模块进行前向传播
# output = attention(target_features, surrounding_features)
#
# # 打印输出结果
# print(output.shape)  # 输出: (batch_size, target_dim + surrounding_dim)