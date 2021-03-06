import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchsummary import summary


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其實這就是一個裁剪的模塊，裁剪多出來的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相當於一個Residual block

        :param n_inputs: int, 輸入通道數
        :param n_outputs: int, 輸出通道數
        :param kernel_size: int, 卷積核尺寸
        :param stride: int, 步長，一般爲1
        :param dilation: int, 膨脹係數
        :param padding: int, 填充係數
        :param dropout: float, dropout比率
        """
        '''think conv1d to tcn constructure~~~'''

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 經過conv1，輸出的size其實是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出來的padding部分，維持輸出時間步爲seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出來的padding部分，維持輸出時間步爲seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        參數初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper給出的TCN結構很好的支持每個時刻爲一個數的情況，即sequence結構，
        對於每個時刻爲一個向量這種一維結構，勉強可以把向量拆成若干該時刻的輸入通道，
        對於每個時刻爲一個矩陣或更高維圖像的情況，就不太好辦。

        :param num_inputs: int， 輸入通道數
        :param num_channels: list，每層的hidden_channel數，例如[25,25,25,25]表示有4個隱層，每層hidden_channel數爲25
        :param kernel_size: int, 卷積核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨脹係數：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            print(in_channels)  # 確定每一層的輸入通道數
            out_channels = num_channels[i]  # 確定每一層的輸出通道數
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        輸入x的結構不同於RNN，一般RNN的size爲(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        這裏把seq_len放在channels後面，把所有時間步的數據拼起來，當做Conv1d的輸入尺寸，實現卷積跨時間步的操作，
        很巧妙的設計。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# conv1 = TemporalConvNet(5, [25, 25, 25]).to(device)
# test_input=torch.ones(1000,5,10000).to(device)
# summary(conv1, (5, 100))
#
# output=conv1(test_input)


test_fletton=nn.Sequential(TemporalConvNet(5, [25, 25, 25])).to(device)
test_input=torch.ones(1000,5,10000).to(device)
output=test_fletton(test_input)






# print(output1.size())


# from torch.autograd import Variable
# input = torch.ones(1, 2, 5, 5)
# input = Variable(input)
# x = torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, groups=1)
# out = x(input)
# print(list(x.parameters())[0].data.numpy().shape)
#
#
# pass


# x = torch.ones(128, 20)
# m = torch.nn.Linear(20, 30)
# output = m(x)
# print(list(m.parameters())[0].data.numpy().shape)
