# nohup  python  training.py   > output_Cora.log 2>&1 &
# nohup  python  training.py   > output_PubMed.log 2>&1 &
# nohup  python  training.py   > output_CiteSeer.log 2>&1 &

import argparse
import os, sys
import time


import torch
import torch.nn as nn
from torch.optim import Adam

from masterModel import ClusteringModel
from utils.data_loading import load_graph_data
from utils.constants import *
import utils.utils as utils
import psgd


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    # num_of_epochs ==  max_loop * max_epoch
    parser.add_argument("--num_of_epochs", type=int, help="total", default=1000*2)
    parser.add_argument("--max_loop", type=int, help="number of training loop", default=100*20)
    parser.add_argument("--max_epoch", type=int, help="number of training epochs", default=1)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    # parser.add_argument("--lr", type=float, help="model learning rate", default=1e-2) # PUBMED
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true',
                        help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training',
                        default=DatasetType.CiteSeer.name)
    parser.add_argument('--train_path', type=str, default="data",
                        help='')
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # parser.add_argument('--save_path', type=str, default="exps/exp", help='Path to save the score and model')
    parser.add_argument('--save_path', type=str, default="exps/paper", help='Path to save the score and model')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--test_step", type=int, help="save model saving (epoch) freq",
                        default=1)

    parser.add_argument('--m', type=float, default=0.05, help='Loss margin in AAM softmax 0.2(0.9801) try 0.35(0.9393)')
    parser.add_argument('--s', type=float, default=0.98, help=' ')
    # parser.add_argument('--s', type=float, default=1, help=' ')
    parser.add_argument('--hdim', type=int, default=11, help=' ')
    parser.add_argument('--nclass', type=int, default=7, help=' ')
    parser.add_argument('--input_dim', type=int, default=1433, help='数据集的特征/模型的输入维度')
    args = parser.parse_args()

    num_of_layers_yzx = 2  # 默认为 2
    hidden_head = 8  # 默认为 8

    #数据集的不同输入特征和类别
    if args.dataset_name.lower() == DatasetType.CORA.name.lower():
        args.input_dim = CORA_NUM_INPUT_FEATURES
        args.nclass = CORA_NUM_CLASSES
    elif args.dataset_name.lower() == DatasetType.CiteSeer.name.lower():
        args.input_dim = CITESEER_NUM_INPUT_FEATURES
        args.nclass = CITESEER_NUM_CLASSES
    elif args.dataset_name.lower() == DatasetType.PubMed.name.lower():
        args.input_dim = PUBMED_NUM_INPUT_FEATURES
        args.nclass = PUBMED_NUM_CLASSES
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported')
    # GAT的模型 参数
    gat_config = {
        "num_of_layers": num_of_layers_yzx,
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": (num_of_layers_yzx - 1) * [hidden_head] + [1],
        # todo 先注释 试试损失函数
        # "num_features_per_layer": [CORA_NUM_INPUT_FEATURES] + (num_of_layers_yzx - 1) * [hidden_head] + [CORA_NUM_CLASSES],
        "num_features_per_layer": [args.input_dim] + (num_of_layers_yzx - 1) * [hidden_head] + [args.hdim],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # base  result is sensitive to dropout
        # "dropout": 0.65,  # 目前最好 yzx__
        # "dropout": 0.70,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    args = vars(args)
    #  一些额外的准备
    args.update(gat_config)
    args = argparse.Namespace(**args)
    args = utils.init_args(args)
    return args

def train_begin(param_arg):
    # Yzx: 一开始默认是 seed 1  其他的均是调试  from 2025年7月8日00:38:00

    # seed 1 cora 84.70% lr 开始 5e-3 add_skip_connection=False
    # seed 1 citeseer 72.30% lr 开始 5e-3 add_skip_connection=False
    # seed 0 pubmed 80.50% lr 开始 1e-2 add_skip_connection=TRUE

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: 加载数据集
    # node_features, node_labels, edge_index, train_index, val_index, test_index = load_graph_data(args, device)
    dataLoader = load_graph_data(vars(param_arg), device)

    # Step 2: 初始化模型
    s = ClusteringModel(**vars(param_arg)).to(device)

    score_file = open(param_arg.score_save_path, "a+")
    metric_file = open(param_arg.metric_save_path, "a+")
    # Step 3: 开始训练

    eps = 0.01
    update_freq = 4
    gamma = 1
    # preconditioner = psgd.KFAC(
    #     s,
    #     eps=eps,
    #     sua=False,
    #     pi=False,
    #     update_freq=update_freq,
    #     # alpha=alpha if alpha is not None else 1.,
    #     alpha=1.,
    #     constraint_norm=False
    # )

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    # Step 4:  验证

    # Step 5: 保存模型  Step 6： 测试:包含各种指标
    loop = 1
    train_max_epo = param_arg.max_epoch
    # 初始化指标
    metric = {}
    metric['Best_ACC'] = [0]
    metric['Best_NMI'] = [0]
    metric['Best_ARI'] = [0]
    metric['Best_purity'] = [0]

    while (loop <= param_arg.max_loop):
        param_arg.max_epoch = train_max_epo
        lam = (float(loop) / float(loop)) ** gamma if gamma is not None else 0.
        loss, lr, train_Metric, train_epoch = s.train_network(dataLoader, param_arg, DataMode.TRAIN, loop, None, lam)

        score_file.write(time.strftime("%m-%d %H:%M:%S") + \
                         " %s, epoch [%5d], Loss: %.4f, LR: %.8f, train ACC: %.2f%%\n" % (
                             DataMode.TRAIN.name, loop * train_epoch, loss, lr, train_Metric[0]))

        if loop % param_arg.test_step == 0:
            sys.stderr.write(" Start Test Time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "  loop: [%d]\n\n" % (loop))
            param_arg.max_epoch = param_arg.test_step  # test_max_epo
            loss, lr, test_Metric, test_epoch = s.train_network(dataLoader, param_arg, DataMode.TEST, loop)
            test_acc = test_Metric[0]
            test_nmi = test_Metric[1]
            test_ari = test_Metric[2]
            test_purity = test_Metric[3]
            score_file.write(time.strftime("%m-%d %H:%M:%S") + \
                             " hdim:%s, %s, [%2d], Loss: %.4f, LR: %.8f, test ACC: %.2f%%\n\n" % (
                                 param_arg.hdim, DataMode.TEST.name, loop * test_epoch, loss, lr, test_acc))
            score_file.flush()
            sys.stderr.write(" End Test Time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n\n")
            metric['Best_ACC'].append(test_acc)
            metric['Best_NMI'].append(test_nmi)
            metric['Best_ARI'].append(test_ari)
            metric['Best_purity'].append(test_purity)
            for k_metric, v_metric in metric.items():
                metric_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                                  " hdim:%s, %s:  now at [%2d], %s at [%2d] is %.2f%%\n\n" % (param_arg.hdim,DataMode.TEST.name, loop * train_epoch, str(k_metric), + \
                    metric[k_metric].index(max(metric[k_metric])) * train_max_epo, + max(max(metric[k_metric]), 0)))
                # if max(metric[k_metric]) == metric[k_metric][-1] and (loop * train_epoch >= 500):
                #     s.save_parameters(config=param_arg, path=param_arg.model_save_path + "/model_%05d.model" % (loop * train_epoch))
                metric_file.flush()
        loop += 1
        sys.stderr.flush()
    sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                     ": 结束训练！！！\n")

# Step 6:


if __name__ == '__main__':

    # Train the graph attention network (GAT)
    args = get_training_args()
    all_range_max_epoch = args.max_epoch

    args.max_epoch = all_range_max_epoch
    sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                     "args.hdim:%s\n" % (args.hdim))
    train_begin(args)

    # # 大批量测试暂停 2025年7月7日22:37:54
    # for i in range(args.nclass, args.nclass+1):
    #     args.hdim = i
    #     args.max_epoch = all_range_max_epoch
    #     sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
    #                      "args.hdim:%s\n" % (i))
    #     train_begin(args)