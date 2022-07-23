from random import random

import torch


from Node import Node, Global_Node
from Args import args_parser
# from DATA_v9 import Data
from DATA_2CLASS import Data
from utils import LR_scheduler, Recorder, Catfish, Summary
from Trainer import train_avg,train_teacher2local_semi,train_normal,train_teacher2local,train_local2teacher

# init args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Running on', args.device)
Data = Data(args)

# Train = Trainer()

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cpu.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cpu.deterministic = True
# # 设置随机数种子
# setup_seed(3407)
# init nodes
# Global_node = Global_Node(Data.test_loader_global, args)
Global_node = Global_Node(Data.test_loader_global, args)
Node_List = [Node(k, Data.train_loader[k], Data.test_loader[k], args) for k in range(args.node_num)]

Catfish(Node_List, args)

Array1 = [[] for i in range(args.node_num)]
Array2 = [[] for i in range(args.node_num)]
Array3 = [[] for i in range(args.node_num)]
Array4 = [[] for i in range(args.node_num)]
Array5 = [[] for i in range(args.node_num)]
Array11,Array22,Array33,Array44= [],[],[],[]
# init variables
recorder = Recorder(args)
Summary(args)
torch.manual_seed(3407)
# start
for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    # LR_scheduler(rounds, Node_List, args)
    weight_accumulator = {}
    # for name, params in Global_node.model.state_dict().items():
    #     weight_accumulator[name] = torch.zeros_like(params)
    if args.algorithm == 'normal':
        for k in range(len(Node_List)):
            print("node{}-----------------------------------------------------------------------------".format(k+1))
            for epoch in range(args.E):
                # if k == 0 or k==1 or k==2:
                #     LR_scheduler2(epoch, Node_List[k], args)
                Node_List[k].fork(Global_node)
                train_normal(Node_List[k])
                recorder.validate(Node_List[k], Array1[k], Array2[k], Array3[k], Array4[k], Array5[k],rounds)
                recorder.printer(Node_List[k])
    elif args.algorithm == 'fed_avg':
        for k in range(len(Node_List)):
            Node_List[k].fork(Global_node)
            train_avg(Node_List[k], Global_node, rounds)
            # print(Node_List[k].model.state_dict().items())
            # for name, _ in Global_node.model.state_dict().items():
            #     weight_accumulator[name].add_(diff[name])
            recorder.validate(Node_List[k], Array1[k], Array2[k], Array3[k], Array4[k], Array5[k],rounds)
            recorder.printer(Node_List[k])
        Global_node.server_aggregate(Global_node,Node_List)
        # Global_node.merge(Global_node,Node_List)
        recorder.Globalvalidate(Global_node,Array11,Array22,Array33,Array44,rounds)
        recorder.printer(Global_node)
    elif args.algorithm == 'fed_mutual_semi':
        for k in range(len(Node_List)):
            if k==3 or k==0 or k==1 or k==4:
                Node_List[k].fork(Global_node)
                train_teacher2local_semi(Node_List[k], Global_node, rounds)
                # print(Node_List[k].model.state_dict().items())
                # for name, _ in Global_node.model.state_dict().items():
                #     weight_accumulator[name].add_(diff[name])
                recorder.validate(Node_List[k], Array1[k], Array2[k], Array3[k], Array4[k], Array5[k], rounds)
                recorder.printer(Node_List[k])
            else:
                Node_List[k].fork(Global_node)
                train_teacher2local(Node_List[k], Global_node, rounds)
                recorder.validate(Node_List[k], Array1[k], Array2[k], Array3[k], Array4[k], Array5[k], rounds)
                recorder.printer(Node_List[k])
                train_local2teacher(Node_List[k], Global_node)

        Global_node.server_aggregate(Global_node,Node_List)
        recorder.Globalvalidate(Global_node,Array11,Array22,Array33,Array44,rounds)
        recorder.printer(Global_node)
    elif args.algorithm == 'fed_mutual':
        for k in range(len(Node_List)):
            Node_List[k].fork(Global_node)
            train_teacher2local(Node_List[k], Global_node,rounds)
            recorder.validate(Node_List[k], Array1[k], Array2[k], Array3[k], Array4[k], Array5[k], rounds)
            recorder.printer(Node_List[k])
            train_local2teacher(Node_List[k], Global_node)
            # print(Node_List[k].model.state_dict().items())
            # for name, _ in Global_node.model.state_dict().items():
            #     weight_accumulator[name].add_(diff[name])
            # recorder.validate(Node_List[k], Array1[k], Array2[k], Array3[k], Array4[k], Array5[k],rounds)
            # recorder.printer(Node_List[k])
        Global_node.server_aggregate(Global_node,Node_List)
        recorder.Globalvalidate(Global_node,Array11,Array22,Array33,Array44,rounds)
        recorder.printer(Global_node)
recorder.finish()
Summary(args)
for k in range(len(Array1)):
    print("node{}-acc:".format(k))
    print(Array1[k])
for k in range(len(Array2)):
    print("node{}-pre:".format(k))
    print(Array2[k])
for k in range(len(Array3)):
    print("node{}-recall:".format(k))
    print(Array3[k])
for k in range(len(Array4)):
    print("node{}-f1:".format(k))
    print(Array4[k])

print("-------------node0--------------")
print("acc:")
print(Array11)
print("pre:")
print(Array22)
print("recall:")
print(Array33)
print("f1:")
print(Array44)

