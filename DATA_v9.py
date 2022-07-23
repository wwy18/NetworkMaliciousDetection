import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

class PCAPset(Dataset):
    def __init__(self, data, label):
        data = data.astype(float)
        label = label.astype(float)
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        self.data = torch.from_numpy(data).unsqueeze(1)
        # self.data = torch.from_numpy(data.values)

        self.labels = torch.from_numpy(label.values).squeeze()

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return self.labels.shape[0]

def Dataset_global():


    x_test_under_sample = pd.read_csv(
        r"IoT-DS2-x-val-1.csv", header=None,
        low_memory=False)
    y_test_under_sample = pd.read_csv(
        r"IoT-DS2-y-val-1.csv", header=None,
        low_memory=False)
    x_test_under_sample = x_test_under_sample.drop(0)
    y_test_under_sample = y_test_under_sample.drop(0)
    # x_test_under_sample = x_test_under_sample.drop(0, axis=1)
    # y_test_under_sample = y_test_under_sample.drop(0, axis=1)

    testset = PCAPset(x_test_under_sample, y_test_under_sample)
    return testset

def Dataset(n):
    trainset, testset = [], []
    new_list = []
    # for i in range(n):
    #     new_list.append(device[i])
    for i in range(n):
        x_train_under_sample = pd.read_csv(
            r"IoT-DS2-x-train-{}.csv".format(i), header=None,
            low_memory=False)
        y_train_under_sample = pd.read_csv(
            r"IoT-DS2-y-train-{}.csv".format(i), header=None,
            low_memory=False)
        x_test_under_sample = pd.read_csv(
            r"IoT-DS2-x-val-{}.csv".format(i), header=None,
            low_memory=False)
        y_test_under_sample = pd.read_csv(
            r"IoT-DS2-y-val-{}.csv".format(i), header=None,
            low_memory=False)
        x_train_under_sample = x_train_under_sample.drop(0)
        y_train_under_sample = y_train_under_sample.drop(0)
        x_test_under_sample = x_test_under_sample.drop(0)
        y_test_under_sample = y_test_under_sample.drop(0)
        trainset1 = PCAPset(x_train_under_sample, y_train_under_sample)
        testset1 = PCAPset(x_test_under_sample, y_test_under_sample)
        trainset.append(trainset1)
        testset.append(testset1)
    return trainset, testset



class Data(object):

    def __init__(self, args):
        self.args = args
        splited_trainset, splited_testset ,splited_testset2= [], [],[]
        splited_trainset, splited_testset = Dataset(args.node_num)

        self.train_loader = [DataLoader(splited_trainset[j], batch_size=args.batchsize, shuffle=True)
                             for j in range(args.node_num)]
        self.test_loader = [DataLoader(splited_testset[j], batch_size=args.batchsize, shuffle=False)
                            for j in range(args.node_num)]

        splited_testset1 = Dataset_global()
        splited_testset2.append(splited_testset1)
        self.test_loader_global = DataLoader(splited_testset1,batch_size=args.batchsize, shuffle=True)
