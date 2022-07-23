import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResDNN1(nn.Module):
    def __init__(self):
        super(ResDNN1, self).__init__()
        self.op1 = nn.Linear(84, 17)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.op1(x)
        return x

class ResDNN2(nn.Module):
    def __init__(self):
        super(ResDNN2, self).__init__()
        self.op1 = nn.Linear(84, 100)
        self.op2 = nn.Linear(100, 17)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.op1(x))
        # x = F.dropout(x, p=0.5)
        x = F.relu(self.op2(x))
        return x

class ResDNN3(nn.Module):
    def __init__(self):
        super(ResDNN3, self).__init__()
        self.op1 = nn.Linear(84, 100)
        self.op2 = nn.Linear(100, 100)
        self.op3 = nn.Linear(100, 17)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.op1(x))
        # x = F.dropout(x, p=0.5)
        x = F.relu(self.op2(x))
        x = F.relu(self.op3(x))
        return x

class ResDNN4(nn.Module):
    def __init__(self):
        super(ResDNN4, self).__init__()
        self.op1 = nn.Linear(84, 100)
        self.op2 = nn.Linear(100, 100)
        self.op3 = nn.Linear(100, 100)
        self.op4 = nn.Linear(100, 17)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.op1(x))
        # x = F.dropout(x, p=0.5)
        x = F.relu(self.op2(x))
        x = F.relu(self.op3(x))
        x = F.relu(self.op4(x))
        return x

# model=ResDNN_pro3()
# summary(model,(1,115))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_block = self.make_layer(1, 8)
        self.fc = nn.Linear(84 * 8, 2)

    def forward(self, x):
        x = self.cnn_block(x)

        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        # return self.sigmoid(x).view(-1)

    def make_layer(self, in_plane, plane):
        return nn.Sequential(nn.Conv1d(in_plane, plane, 3, 1, 1),
                             nn.BatchNorm1d(plane),
                             nn.ReLU())

model=CNN()
summary(model,(1,84))

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.cnn_block = self.make_layer(1, 5)
        self.fc = nn.Linear(84 * 5, 17)

    def forward(self, x):
        x = self.cnn_block(x)

        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        # return self.sigmoid(x).view(-1)

    def make_layer(self, in_plane, plane):
        return nn.Sequential(nn.Conv1d(in_plane, plane, 3, 1, 1),
                             nn.BatchNorm1d(plane),
                             nn.ReLU())
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.cnn_block = self.make_layer(1, 5)
        self.fc1 = nn.Linear(84 * 5, 16*5)
        self.fc2 = nn.Linear(16 * 5, 17)

    def forward(self, x):
        x = self.cnn_block(x)

        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        # return self.sigmoid(x).view(-1)

    def make_layer(self, in_plane, plane):
        return nn.Sequential(nn.Conv1d(in_plane, plane, 3, 1, 1),
                             nn.BatchNorm1d(plane),
                             nn.ReLU())

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        self.cnn_block = self.make_layer(1, 5)
        self.fc1 = nn.Linear(84 * 5, 16*5)
        self.fc2 = nn.Linear(16 * 5, 16 * 2)
        self.fc3 = nn.Linear(16 * 2, 17)

    def forward(self, x):
        x = self.cnn_block(x)

        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        # return self.sigmoid(x).view(-1)

    def make_layer(self, in_plane, plane):
        return nn.Sequential(nn.Conv1d(in_plane, plane, 3, 1, 1),
                             nn.BatchNorm1d(plane),
                             nn.ReLU())

# 多加了一个线性层的CNN
class CNN_pro(nn.Module):
    def __init__(self):
        super(CNN_pro, self).__init__()
        self.cnn_block = self.make_layer(1, 16)

        self.fc = nn.Linear(84 * 16, 64 * 16)
        self.fc2 = nn.Linear(64 * 16, 17)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn_block(x)

        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.fc2(x)
        return x

    #         return self.sigmoid(x).view(-1)

    def make_layer(self, in_plane, plane):
        return nn.Sequential(nn.Conv1d(in_plane, plane, 3, 1, 1),
                             nn.BatchNorm1d(plane),
                             nn.ReLU())

model=CNN_pro()
summary(model,(1,84))
