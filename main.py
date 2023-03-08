import argparse
import os
import random
import sys
from copy import deepcopy

from fedlab.utils import SerializationTool
from fedlab.utils.functional import evaluate
from torch import nn
from tqdm import tqdm

from trainers import SerialTrainer, SubsetSerialTrainer
from FMLP_Rec.datasets import FMLPRecDataset

sys.path.append("C:\\Users\\80623\Desktop\\federated_architecture_\\FMLP_Rec")
import torch
from munch import Munch

from FMLP_Rec.models import FMLPRecModel
from FMLP_Rec.utils import get_dataloder, get_seq_dic, check_path, set_seed, get_local_time
from fedlab.utils.aggregator import Aggregators


def main():
    '''
    设置超参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)

    # model args
    parser.add_argument("--model_name", default="FMLPRec", type=str)
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--no_filters", action="store_true",
                        help="if no filters, filter layers transform to self-attention")

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--full_sort", action="store_true")
    parser.add_argument("--patience", default=10, type=int,
                        help="how long to wait after last time validation loss improved")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    args = parser.parse_args()
    '''
    设置和联邦学习相关的参数
    '''
    federated_args = Munch
    federated_args.total_client = 10
    federated_args.sample_ratio = 0.7
    federated_args.alpha = 0.5
    federated_args.seed = 42
    federated_args.preprocess = True
    federated_args.cuda = False
    federated_args.com_round = 10

    # FL settings
    # 每轮参与训练的客户数量
    num_per_round = int(federated_args.total_client * federated_args.sample_ratio)
    # 设置聚合器
    aggregator = Aggregators.fedavg_aggregate
    # 参与联邦学习的客户数量
    total_client_num = federated_args.total_client

    set_seed(args.seed)
    # 检查路径是否合法
    check_path(args.output_dir)
    # 检查cuda是否可用
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    # 获得训练、验证、测试集对应的用户序列数据 和 最大的 item_id
    seq_dic, max_item = get_seq_dic(args)
    args.item_size = max_item + 1
    # 设置训练集
    train_dataset = FMLPRecDataset(args, seq_dic['user_seq'], data_type='train')
    # 设置测试集
    test_dataset = FMLPRecDataset(args, seq_dic['user_seq_test'], data_type='test')

    cur_time = get_local_time()
    # 设置模型名称
    args_str = f'{args.model_name}-{args.data_name}-{cur_time}'
    # 设置日志存储位置
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    '''
        处理数据集 
    '''
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    model = FMLPRecModel(args=args)
    # 将客户与数据对应
    data_indices = []
    for indice in range(0,max_item-1):
        data_indices.append([indice])
    # 本地模型
    local_model = deepcopy(model)
    #在一个进程中训练多个客户。
    trainer = SubsetSerialTrainer(model = local_model, dataset=train_dataset,
                                           data_slices=data_indices,
                                           aggregator=aggregator,
                                           cuda=False,
                                           args={
                                               "batch_size": args.batch_size,
                                               "epochs": args.epochs,
                                               "lr": args.lr,
                                               "log_file":args.log_file
                                           })


#     train procedure
    to_select = [i for i in range(total_client_num)]
    for round in range(federated_args.com_round):
        # 获得服务器模型的序列化参数,展开模型每层的参数张量并且串联成一个
        model_parameters = SerializationTool.serialize_model(model)
        # 随机选取几个客户机进行训练
        selection = random.sample(to_select, num_per_round)
        # 获得这些客户机训练完成后的聚合的参数
        aggregated_parameters = trainer.train(model_parameters=model_parameters,
                                              id_list=selection,
                                              aggregate=True)
        # 将聚合的序列化参数更新给服务器模型
        SerializationTool.deserialize_model(model,aggregated_parameters)
        # 对当前模型效果进行评估
        res = trainer.c_evaluate(round,test_dataloader)
        print("loss:",res)



main()



