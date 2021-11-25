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


# net = EEG_Net_Bin().to(device)
# summary(net, (1, 16, 1000))
#
class Sigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


net = nn.Sequential(TemporalConvNet(16, [25, 25, 25]).to(device)
                    , Flatten()
                    , nn.Linear(25000, 250)
                    , nn.ReLU()
                    , nn.Linear(250, 2)
                    ).to(device)
summary(net, (16, 1000))
pass


def loss_function(inputs, targets):
    return nn.CrossEntropyLoss()(inputs, targets)


def evaluate(model, X, Y, params=["acc"]):
    results = []
    inputs = Variable(torch.from_numpy(X).to(device))
    predicted = model(inputs)

    predicted = predicted.data.cpu().numpy().argmax(1)

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2 * precision * recall / (precision + recall))
    return results


criterion_loss = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_acc = []
test_acc = []
batch_size = 15

for epoch in range(1000):  # loop over the dataset multiple times
    print("\nEpoch ", epoch)
    kkk = len(X_train) / batch_size - 1
    running_loss = 0.0
    for i in range(int(kkk)):
        s = i * batch_size
        e = i * batch_size + batch_size

        inputs = torch.from_numpy(X_train[s:e])
        labels = torch.LongTensor(np.array([y_train[s:e]]).T * 1.0)

        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # print(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs.size())

        # print(torch.max(labels, 1)[0])
        # loss = criterion(outputs, labels)
        loss = criterion_loss(outputs, labels.squeeze())
        print(labels.squeeze())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 2000 == 1999:  # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    # Validation accuracy
    params = ["acc", "auc", "fmeasure"]
    print(params)
    print("Training Loss ", running_loss)
    print("Train - ", evaluate(net, X_train, y_train, params))
    train_acc.append(evaluate(net, X_train, y_train, params)[0])
    # print(train_acc)

    # print("Validation - ", evaluate(net, X_val, y_val, params))
    print("Test - ", evaluate(net, X_test, y_test, params))
    test_acc.append(evaluate(net, X_test, y_test, params)[0])
    # print(test_acc)
    util.checkdir_exist(os.path.join(os.getcwd(), 'torch_model'))
    if evaluate(net, X_test, y_test, params)[0] >= 0.6:

        torch.save(net, os.path.join(os.getcwd(), 'torch_model', 'testmodel.pt'))



        print('model save')

    plt.figure(1)
    plt.clf()

    # plt.plot(N, self.losses, label = "train_loss")
    plt.plot(train_acc, label="train_acc")
    # plt.plot(N, self.val_losses, label = "val_loss")
    plt.plot(test_acc, label="test_acc")
    plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")

    plt.legend()
    # Make sure there exists a folder called output in the current directory
    # or replace 'output' with whatever direcory you want to put in the plots
    # Make sure there exists a folder called output in the current directory
    # or replace 'output' with whatever direcory you want to put in the plots
    plt.pause(0.01)

