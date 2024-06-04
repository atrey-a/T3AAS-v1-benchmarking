import torch
import torch.nn as nn


class TwoLevelGRU(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_classes,
        num_descriptors=5,
        dropout_rate=0.5,
        hidden_size_1=512,
        hidden_size_2=128,
        hidden_size_3=512,
        hidden_size_4=128,
        layer1_bidirectional=True,
        layer2_bidirectional=True,
    ):
        super(TwoLevelGRU, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.GRU(
                    num_descriptors, hidden_size_1, bidirectional=layer1_bidirectional, dropout=0.5
                )
                for i in range(self.num_columns)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size_1 * (int(layer1_bidirectional) + 1),
                    out_features=hidden_size_2,
                )
                for i in range(self.num_columns)
            ]
        )

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)

        self.layer2 = nn.GRU(
            hidden_size_2 * self.num_columns,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
            dropout=0.5
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):

        layer1_out = []

        for i in range(self.num_columns):
            t = x[:, :, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)
        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class TwoLevelGRU_(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_classes,
        num_descriptors=5,
        dropout_rate=0.5,
        hidden_size_1=512,
        hidden_size_2=128,
        hidden_size_3=512,
        hidden_size_4=128,
        layer1_bidirectional=True,
        layer2_bidirectional=True,
    ):
        super(TwoLevelGRU_, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.GRU(num_columns, hidden_size_1, bidirectional=layer1_bidirectional)
                for _ in range(self.num_descriptors)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size_1 * (int(layer1_bidirectional) + 1), hidden_size_2
                )
                for _ in range(self.num_descriptors)
            ]
        )

        self.layer2 = nn.GRU(
            hidden_size_2 * self.num_descriptors,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):
        layer1_out = []

        for i in range(self.num_descriptors):
            t = x[:, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)

        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class TwoLevelLSTM(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_classes,
        num_descriptors,
        dropout_rate=0.5,
        hidden_size_1=512,
        hidden_size_2=128,
        hidden_size_3=512,
        hidden_size_4=128,
        layer1_bidirectional=True,
        layer2_bidirectional=True,
    ):
        super(TwoLevelLSTM, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.LSTM(
                    num_descriptors, hidden_size_1, bidirectional=layer1_bidirectional
                )
                for i in range(self.num_columns)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size_1 * (int(layer1_bidirectional) + 1),
                    out_features=hidden_size_2,
                )
                for i in range(self.num_columns)
            ]
        )

        self.layer2 = nn.LSTM(
            hidden_size_2 * num_columns,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):
        layer1_out = []

        for i in range(self.num_columns):
            t = x[:, :, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)

        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class TwoLevelLSTM_(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_classes,
        num_descriptors=5,
        dropout_rate=0.5,
        hidden_size_1=512,
        hidden_size_2=128,
        hidden_size_3=512,
        hidden_size_4=128,
        layer1_bidirectional=True,
        layer2_bidirectional=True,
    ):
        super(TwoLevelLSTM_, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.LSTM(num_columns, hidden_size_1, bidirectional=layer1_bidirectional)
                for _ in range(self.num_descriptors)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size_1 * (int(layer1_bidirectional) + 1), hidden_size_2
                )
                for _ in range(self.num_descriptors)
            ]
        )

        self.layer2 = nn.LSTM(
            hidden_size_2 * num_descriptors,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):
        layer1_out = []

        for i in range(self.num_descriptors):
            t = x[:, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)

        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x
