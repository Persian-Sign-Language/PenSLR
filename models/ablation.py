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
    

class AblationModel(nn.Module):
    def __init__(self, inp, out, ablation):
        super().__init__()
        self.has_cnn = ablation.has_cnn
        self.num_batch_norm = ablation.num_batch_norm
        if ablation.has_cnn:
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ablation.kernel_size, stride=ablation.stride, padding=ablation.padding)
            h_size = int(1 + (inp + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) / self.conv.stride[0])
            self.bn = nn.BatchNorm1d(h_size)
        else:
            h_size = inp
        
        # LSTMS and Dropouts
        self.lstm_module = [
            nn.LSTM(h_size, 32 * 2**(2 - ablation.num_lstm), 1, batch_first=True, bidirectional=ablation.bidirectional),
            nn.Dropout(p=0.3)
        ]
        if ablation.num_lstm == 2:
            self.lstm_module.append(nn.LSTM(32 * (int(ablation.bidirectional) + 1), 64, 1, batch_first=True, bidirectional=ablation.bidirectional))
            self.lstm_module.append(nn.Dropout(p=0.3))
            
        self.lstm_module = nn.ModuleList(self.lstm_module)
        self.non_linearity = nn.Tanh()

        # Linears and Batch Norms
        self.mlp_module = nn.ModuleList([
            nn.Linear(64 * (int(ablation.bidirectional) + 1), 64), 
            nn.Linear(64, 32), 
            nn.Linear(32, out + 1)]
        )
        self.bns = []
        if ablation.num_batch_norm > 0:
            self.bns.append(nn.BatchNorm1d(64))
        if ablation.num_batch_norm > 1:
            self.bns.append(nn.BatchNorm1d(32))
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        if self.has_cnn:
            x = x.transpose(2, 1).unsqueeze(1)
            x = self.conv(x)
            x = x.squeeze(1)
            x = self.bn(x)
            x = x.transpose(2, 1)
        for module in self.lstm_module:
            if isinstance(module, nn.LSTM):
                x, _ = module(x)
            else:
                x = module(x)
        for i in range(2):
            x = self.mlp_module[i](x)
            x = self.non_linearity(x)
            if i < self.num_batch_norm:
                x = x.permute(0, 2, 1)
                x = self.bns[i](x)
                x = x.permute(0, 2, 1)
        x = self.mlp_module[-1](x)
        return x.transpose(0, 1)


