import copy
import torch
from torchvision import models
import torch.nn as nn
import Model_v2
from Args import args_parser
args = args_parser()
def init_model(model_type):
    model = []
    if model_type == 'ResDNN1':
        model = Model_v2.ResDNN1()
        print("DNN1")
    elif model_type == 'ResDNN2':
        model = Model_v2.ResDNN2()
        print("DNN2")
    elif model_type == 'ResDNN3':
        model = Model_v2.ResDNN3()
        print("DNN3")
    elif model_type == 'ResDNN4':
        model = Model_v2.ResDNN4()
        print("DNN4")
    elif model_type == 'CNN':
        model = Model_v2.CNN()
        print("CNN")
    elif model_type == 'CNN_pro':
        model = Model_v2.CNN_pro()
        print("CNN_pro")
    elif model_type == 'CNN2':
        model = Model_v2.CNN2()
        print("CNN2")
    elif model_type == 'CNN3':
        model = Model_v2.CNN3()
        print("CNN3")
    elif model_type == 'CNN4':
        model = Model_v2.CNN4()
        print("CNN4")
    elif model_type == 'resnet':
        resnet18 = models.resnet18()
        # 修改全连接层的输出
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 10)
        model = resnet18
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, train_data, test_data, args):
        self.args = args
        self.num = num
        self.device = self.args.device
        self.train_data = train_data
        self.test_data = test_data
        self.model = init_model(self.args.local_model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
        self.meme = init_model(self.args.global_model).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)
    def fork(self, global_node):
        self.meme = copy.deepcopy(global_node.model).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = init_model(self.args.global_model).to(self.device)
        self.test_data = test_data
        self.Dict = self.model.state_dict()


    def server_aggregate(self,Global_node, Node_List):
        weights_zero(self.model)
        client_lens = []
        next_global_dict = dict()
        for name, data in Global_node.model.state_dict().items():
            next_global_dict[name] = torch.zeros_like(data)
        for i in range(len(Node_List)):
            datalength = len(Node_List[i].train_data)
            client_lens.append(datalength)
        total = sum(client_lens)
        n = len(Node_List)
        # n = num_selected
        global_dict = Global_node.model.state_dict()

        for name, data in Global_node.model.state_dict().items():
            for i in range(len(Node_List)):
                next_global_dict[name] += Node_List[i].meme.state_dict()[name]
            next_global_dict[name] = next_global_dict[name] / 5
        Global_node.model.load_state_dict(next_global_dict)

