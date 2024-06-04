import torch
import torch.nn as nn


class SlitCNN_with_LSTM(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes, lstm_hidden_size=256):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.factor = int(self.num_rows / 512)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30*self.factor,self.num_columns),stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8*self.factor,1),stride=1)

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)

        self.norm = nn.LayerNorm([64,self.num_rows-(30*self.factor)+1,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.Linear(in_features=lstm_hidden_size*int((self.num_rows+2-(38*self.factor))*2),out_features=128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self,x):
        out = torch.unsqueeze(x,dim=1)
        out = self.conv1(out)
        out = self.lrelu(out)
        out = self.norm(out)
        out = self.conv2(out)
        out = self.lrelu(out)
        out = out.squeeze(-1)
        out = out.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TwoStream_SlitCNN_with_LSTM_Parallel(nn.Module):
    def __init__(self, num_rows, num_columns, num_classes, lstm_hidden_size=256):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])
        self.lstm = nn.ModuleList([ 
            nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)
            for i in range(2)
        ])
        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=int((self.num_rows-36))*self.lstm_hidden_size*2,out_features=128)
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
            t = t.squeeze(-1)
            t = t.permute(0, 2, 1)
            t, (hn, cn) = self.lstm[i](t)
            t = self.flatten(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out

class TwoStream_SlitCNN_with_LSTM_Sequential(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes, lstm_hidden_size=256):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)

        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=476*2*lstm_hidden_size,out_features=128)
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
            t = t.squeeze(-1)
            t = t.permute(0, 2, 1)
            t, (hn, cn) = self.lstm(t)
            t = self.flatten(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out
