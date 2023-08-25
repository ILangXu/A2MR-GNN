import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, num_features):
        super(SimpleAttention, self).__init__()
        self.attention_linear1 = nn.Linear(num_features, num_features)
        self.attention_linear2 = nn.Linear(num_features, 1)

    def forward(self, x_r1, x_r2):
        # Concatenate x_r1 and x_r2 along feature dimension
        x_concat = torch.stack([x_r1, x_r2], dim=1)  # Shape: (num_nodes, 2, num_features)

        # Calculate attention scores
        x_hidden = F.relu(self.attention_linear1(x_concat))  # Shape: (num_nodes, 2, num_features)
        attention_scores = torch.sigmoid(self.attention_linear2(x_hidden))  # Shape: (num_nodes, 2, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (num_nodes, 2, 1)

        # Weighted sum
        x_weighted = x_concat * attention_weights  # Shape: (num_nodes, 2, num_features)
        x_output = x_weighted.sum(dim=1)  # Shape: (num_nodes, num_features)

        return x_output
