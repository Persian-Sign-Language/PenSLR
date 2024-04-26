import torch.nn as nn
import torch
    
class M1(nn.Module):
    def __init__(self, inp, out, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        h_size = int(1 + (inp + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) / self.conv.stride[0])
        # print(h_size)
        self.bn = nn.BatchNorm1d(h_size)
        self.rnn1 = nn.LSTM(h_size, 32, 1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.rnn2 = nn.LSTM(64, 64, 1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, out + 1)
    def forward(self, x):
        x_conv = x.transpose(2, 1).unsqueeze(1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.squeeze(1)
        x_conv = self.bn(x_conv)
        x_conv = x_conv.transpose(2, 1)
        # x_conv = nn.ReLU()(x_conv)
        out, _ = self.rnn1(x_conv)
        out = nn.Tanh()(out)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = nn.Tanh()(out)
        out = self.dropout2(out)
        out = self.linear1(out)
        out = nn.Tanh()(out)
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        out = out.permute(0, 2, 1)
        out = self.linear2(out)
        out = nn.Tanh()(out)
        out = self.linear3(out)
        out = out.transpose(0, 1)
        return out
    
class M2(nn.Module):
    def __init__(self, inp, out, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        h_size = int(1 + (inp + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) / self.conv.stride[0])
        # print(h_size)
        self.bn = nn.BatchNorm1d(h_size)
        self.rnn1 = nn.LSTM(h_size, 32, 1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.rnn2 = nn.LSTM(64, 64, 1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)   
        self.linear3 = nn.Linear(32, out + 1)
    def forward(self, x):
        ########## Feature Fusion
        x_conv = x.transpose(2, 1).unsqueeze(1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.squeeze(1)
        x_conv = self.bn(x_conv)
        x_conv = x_conv.transpose(2, 1)
        # x_conv = nn.LeakyReLU()(x_conv)
        ########## Temporal Feature Extraction
        # print(x_conv.shape)
        out, _ = self.rnn1(x_conv)
        out = nn.Tanh()(out)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = nn.Tanh()(out)
        out = self.dropout2(out)
        ########## Sequence Generation Layer
        out = self.linear1(out)
        out = nn.Tanh()(out)
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        out = out.permute(0, 2, 1)
        out = self.linear2(out)
        out = nn.Tanh()(out)
        out = out.permute(0, 2, 1)
        out = self.bn3(out)
        out = out.permute(0, 2, 1)
        out = self.linear3(out)
        out = out.transpose(0, 1)
        return out
    

if __name__ == "__main__":
    model = M2(17, 23)
    x = torch.rand(2, 31, 17)
    print(model(x).shape)