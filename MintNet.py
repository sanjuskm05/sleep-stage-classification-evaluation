import torch
import torch.nn as nn
import torch.nn.functional as F

class MintNet(nn.Module):
    def __init__(self, n=6):
        """
        TODO : documents
        """
        super(MintNet, self).__init__()
        '''
        Representation layer (initialization)

        '''
        Fs = 100
        self.conv_1 = nn.Conv1d(in_channels=n, out_channels=64, kernel_size=int(Fs / 2), stride=int(Fs / 16), padding=int(Fs / 4) -1)
        self.pool_1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.droput_1 = nn.Dropout(p=.5)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, padding=4)
        self.conv_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=4)
        self.conv_4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=4)
        self.pool_2 = nn.MaxPool1d(kernel_size=4, stride=4)

        '''
        Representation layer (Fine-tuning)

        '''
        self.conv_1_ft = nn.Conv1d(in_channels=n, out_channels=32, kernel_size=int(Fs * 4), stride=int(Fs / 2), padding=int(Fs*2) -1)
        self.pool_1_ft = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv_2_ft = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=6, padding=3)
        self.conv_3_ft = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, padding=3)
        self.conv_4_ft = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, padding=3)
        self.pool_2_ft = nn.MaxPool1d(kernel_size=2, stride=2)

        '''
        TODO - Some reshaping, We are not sure :)
        '''
        self.lstm_1 = nn.LSTM(input_size=1, hidden_size=512, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=int(512*2), hidden_size=1, num_layers=1, bidirectional=False)

        '''
        Fully Connected
        '''

        self.fc = nn.Linear(in_features=3200, out_features=6)

    def forward(self, input, dev):
        input = input.permute(0,2,1).to(dev)
        #input = input.unsqueeze(dim=-1).permute(0,2,1).to(dev)
        #print(input.shape)
        x = self.conv_1(input)
        x = F.relu(x)
        x = self.pool_1(x)
        x = self.droput_1(x)
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = self.pool_2(x)
        x = torch.flatten(x, start_dim=1)

        x_hat = self.conv_1_ft(input)
        x_hat = F.relu(x_hat)
        x_hat = self.pool_1_ft(x_hat)
        x_hat = self.droput_1(x_hat)
        x_hat = F.relu(self.conv_2_ft(x_hat))
        x_hat = F.relu(self.conv_3_ft(x_hat))
        x_hat = F.relu(self.conv_4_ft(x_hat))
        x_hat = self.pool_2_ft(x_hat)
        x_hat = torch.flatten(x_hat, start_dim=1)
        merged_layers = torch.cat((x, x_hat), dim=-1)
        out_hat = self.droput_1(merged_layers)
        '''
        TODO some reshaping required
        '''

        out = out_hat.unsqueeze(dim=-1)
        out, hidden = self.lstm_1(out)
        out = self.droput_1(out)
        out, hidden = self.lstm_2(out)
        out = self.droput_1(out)

        # fc_input = out_hat.unsqueeze(dim=-1)
        #
        # out_hat = self.fc(fc_input)#correct this out must be from after merged_layer
        #
        # merged_layers = torch.cat((out, out_hat), dim=-1)

        out = self.fc(out.squeeze(dim=-1))
        out = self.droput_1(out)
        out = F.softmax(out, dim=0)

        return out

