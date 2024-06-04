import torch
import torch.nn as nn


class CNN1D(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes):
        super(CNN1D, self).__init__()

        self.num_columns = num_columns
        self.num_rows = num_rows

        self.conv1d_cnn_1 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.conv1d_cnn_2 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        self.conv1d_cnn_3 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.conv1d_cnn_4 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        self.conv1d_cnn_5 = nn.Conv1d(
            self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
        )

        self.final_layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(int(self.num_rows / 32) * self.num_columns, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x = self.conv1d_cnn_1(x)
        x = self.conv1d_cnn_2(x)
        x = self.conv1d_cnn_3(x)
        x = self.conv1d_cnn_4(x)
        x = self.conv1d_cnn_5(x)

        x = self.final_layer(x)

        return x
