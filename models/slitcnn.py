import torch
import torch.nn as nn


class OneStreamSlitCNN(nn.Module):
    def __init__(self, num_rows, num_columns, num_classes):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.factor = int(self.num_rows / 512)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30*self.factor,self.num_columns),stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8*self.factor,1),stride=1)

        self.norm = nn.LayerNorm([64,self.num_rows-(30*self.factor)+1,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.Linear(in_features=128*int((self.num_rows+2-(38*self.factor))/2),out_features=128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self,x):
        x = torch.unsqueeze(x,dim=1)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TwoStreamSlitCNN(nn.Module):
    def __init__(self, num_rows, num_columns, num_classes):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes

        if not self.num_columns == 6:
            raise Exception("Invalid dataloader!")

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])

        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=128*int((self.num_rows-36)/2),out_features=128)
            for i in range(2)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(2):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = self.maxpool(t)
            t = self.flatten(t)
            t = torch.squeeze(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out
