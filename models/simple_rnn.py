import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):

    def __init__(
        self,
        num_rows,
        num_columns,
        num_classes,
        hidden_size_1=512,
        hidden_size_2=128,
        bidirectional=True,
    ):
        super(SimpleLSTM, self).__init__()

        self.layer1 = nn.LSTM(num_columns, hidden_size_1, bidirectional=bidirectional, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size_1 * (int(bidirectional) + 1), hidden_size_2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc2 = nn.Linear(num_rows * hidden_size_2, num_classes)

    def forward(self, x):

        x, _ = self.layer1(x)
        x = self.fc1(x)
        x = self.flatten(x)
        x = self.fc2(x)

        return x


class SimpleGRU(nn.Module):

    def __init__(
        self,
        num_rows,
        num_columns,
        num_classes,
        hidden_size_1,
        hidden_size_2,
        bidirectional,
    ):
        super(SimpleGRU, self).__init__()

        self.layer1 = nn.GRU(num_columns, hidden_size_1, bidirectional=bidirectional, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size_1 * (int(bidirectional) + 1), hidden_size_2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc2 = nn.Linear(num_rows * hidden_size_2, num_classes)

    def forward(self, x):

        x, _ = self.layer1(x)
        x = self.fc1(x)
        x = self.flatten(x)
        x = self.fc2(x)

        return x
