import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
from torch.autograd import Variable
from torchsummary import summary
from model_network.AI_model_torch import EEG_Net_Bin, TemporalConvNet, Flatten
import os
from util import Filter, util

from dataprocessing import Load, Trial_Cannel_data

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = 3
end = 5
step = 0.05
"""
S01 add S03 2s~5s step=0.05 OK ,but step=0.02 training loss =nan
# It seeem to be some problem in different sampling rate
"""

sub_dic = {"S01": 500, "S02": 500, "S03": 512, "S04": 512, "S05": 512, "S10": 500}
sub_list = []
for xx in sub_dic:
    sub_list.append(xx)

sub_num = sub_list[1]  # /////////subject
modelpath = os.path.join(os.getcwd(), sub_num, "model")
file_path = util.returnpath([sub_num])
left_eegdata, right_eegdata, left_eegdata_test, right_eegdata_test = Load(
    file_path, sub_dic, start, end, step
)

XT, YT = Trial_Cannel_data(left_eegdata, right_eegdata)
Xt, Yt = Trial_Cannel_data(left_eegdata_test, right_eegdata_test)

# X_train = np.float32(np.expand_dims(XT, 1))
# X_test = np.float32(np.expand_dims(Xt, 1))
X_train = np.float32(XT)
X_test = np.float32(Xt)
y_train = np.float32(np.squeeze(YT))
y_test = np.float32(np.squeeze(Yt))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# 建構LSTM物件

# 宣告反向傳播機制
loss_fnc = nn.CrossEntropyLoss()


# 宣告損失函數

class LSTM(nn.Module):
    '''
    input:(batch_size,feature,time)
    after this LSTM class nn.Linear(hidden_feature, num_class)
    '''

    def __init__(self, in_feature=28, hidden_feature=100, num_layers=2, class_num=2, batch_size=5):
        super(LSTM, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.num_layers = num_layers
        self.class_num = class_num
        self.batch_size = batch_size

        self.rnn = nn.LSTM(self.in_feature, self.hidden_feature, self.num_layers, bidirectional=True)  # 使用兩層lstm
        self.Linear = nn.Linear(self.hidden_feature*2, self.class_num)
        self.h = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_feature).to(device), torch.zeros(
            self.num_layers*2, self.batch_size, self.hidden_feature).to(device))

    def forward(self, x, **kwargs):
        x = x.permute(2, 0, 1)  # 將最後一維放到第一維，變成（batch，28,28）
        x, self.h = self.rnn(x, self.h)
        print(x.size())



        x = x[-1, :, :]


        x = self.Linear(x)
        return x, self.h


#
net = LSTM(in_feature=16, hidden_feature=20, num_layers=2, batch_size=int(X_train.shape[0])).to(device)
test_input = torch.from_numpy(X_train).to(device)

print(test_input.size())

x = net(test_input)

print(net)

# class EasyLSTM(nn.LSTM):
#
#     def __init__(self, *args, **kwargs):
#         nn.LSTM.__init__(self, *args, **kwargs)
#         self.num_direction = 1 + self.bidirectional
#         state_size = (self.num_layers * self.num_direction, 1, self.hidden_size)
#         self.init_h = nn.Parameter(torch.zeros(state_size))
#         self.init_c = nn.Parameter(torch.zeros(state_size))
#
#     def forward(self, rnn_input, prev_states=None):
#         rnn_input = rnn_input.permute(2, 0, 1)
#         batch_size = rnn_input.size(1)
#         if prev_states is None:
#             state_size = (self.num_layers * self.num_direction, batch_size, self.hidden_size)
#             init_h = self.init_h.expand(*state_size).contiguous()
#             init_c = self.init_c.expand(*state_size).contiguous()
#             prev_states = (init_h, init_c)
#         rnn_output, states = nn.LSTM.forward(self, rnn_input, prev_states)
#         return rnn_output, states
#
#
# class LSTM_main_class(nn.Module):
#
#     def __init__(self, in_feature=28, hidden_feature=100, num_layers=1, class_num=2, batch_size=5, *args, **kwargs):
#         super(LSTM_main_class, self).__init__()
#
#         self.in_feature = in_feature
#         self.hidden_feature = hidden_feature
#         self.num_layers = num_layers
#         self.class_num = class_num
#         self.batch_size = batch_size
#         self.Linear = nn.Linear(self.hidden_feature, self.class_num)
#         self.net=EasyLSTM(input_size=self.in_feature, hidden_size=self.hidden_feature, num_layers=num_layers).to(device)
#
#     def forward(self, x, init_flag=None):
#         if init_flag != None:
#
#             x, self.state = self.net(x)
#         else:
#             x, self.state = self.net(x,self.state)
#         x = self.Linear(x)
#         return x
#
# net= LSTM_main_class(in_feature=16, hidden_feature=100, num_layers=2, class_num=2, batch_size=5).to(device)
#
# test_input = torch.from_numpy(X_train[:50]).to(device)
# test_input2 = torch.from_numpy(X_train[50:100]).to(device)
#
# x= net(test_input,True)
#
#
# pass


# net = EasyLSTM(input_size=16, hidden_size=20,num_layers=2).to(device)
# test_input = torch.from_numpy(X_train[:50]).to(device)
# test_input2 = torch.from_numpy(X_train[50:100]).to(device)
#
# x, state = net(test_input)
# print(x[-1] == state[0])
# x, _ = net(test_input2, state)
#
# print(x)
#
#
# def test_kawg(n, ii=6, jj=3, *args, **kwargs):
#     print(n)
#
#     kwargs.setdefault('aa', 8)
#     kwargs.setdefault('pp', 9)
#     print(kwargs)
#
#     for a, b in kwargs.items():
#         print(a, b)
#
#
# test_kawg(3, 6, 8, '6', 8, aa=6)
