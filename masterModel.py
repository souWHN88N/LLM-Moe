
import time, sys

# import infomap
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.constants import DataMode, LayerType
# from torch_geometric.nn import GCNConv
from layer import GCNConv
from utils.utils import get_training_state
from loss import Control_Contrastive, KNN_Loss, AAMsoftmax
import sklearn.metrics as metrics
import math, numpy as np
from scipy.sparse import coo_matrix


class CRD(nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class ssp_GCN_Model(nn.Module):
    def __init__(self, input_dim=1433, hid_dim=16, out_dim=7, dropout=0.5):
        super(ssp_GCN_Model, self).__init__()
        self.crd = CRD(input_dim, hid_dim, dropout)
        self.cls = CLS(hid_dim, out_dim)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, x, adj):
        x = self.crd(x, adj, None)
        x = self.cls(x, adj, None)
        return x


# class GCN(nn.Module):
#     def __init__(self, input_dim, hid_dim, out_dim, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(input_dim, hid_dim)
#         self.gc2 = GraphConvolution(hid_dim, out_dim)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = F.relu(x)
#         return x


# class GAT_InfoMap(nn.Module):
#     def __init__(self, input_dim, hid_dim, out_dim, dropout):
#         super(GAT_InfoMap, self).__init__()
#
#     def forward(self, x, adj):
#         x = self.cluster_by_infomap(x)
#         return x
#
#     def get_links(self, single, links, nbrs, dists):
#         for i in tqdm(range(nbrs.shape[0])):
#             count = 0
#             for j in range(0, len(nbrs[i])):
#                 # 排除本身节点
#                 if i == nbrs[i][j]:
#                     pass
#                 elif dists[i][j] <= 1 - 0.58: # min_sim = 0.58
#                     count += 1
#                     links[(i, nbrs[i][j])] = float(1 - dists[i][j])
#                 else:
#                     break
#             # 统计孤立点
#             if count == 0:
#                 single.append(i)
#         return single, links
#     def cluster_by_infomap(self, nbrs, dists, pred_label_path, save_result=False):
#         """
#         基于infomap的聚类
#         :param nbrs:
#         :param dists:
#         :param pred_label_path:
#         :return:
#         """
#         single = []
#         links = {}
#
#         single, links = self.get_links(single=single, links=links, nbrs=nbrs, dists=dists)
#
#         infomapWrapper = infomap.Infomap("--two-level --directed")
#         for (i, j), sim in tqdm(links.items()):
#             _ = infomapWrapper.addLink(int(i), int(j), sim)
#
#         # 聚类运算
#         infomapWrapper.run()
#
#         label2idx = {}
#         idx2label = {}
#
#         # 聚类结果统计
#         for node in infomapWrapper.iterTree():
#             # node.physicalId 特征向量的编号
#             # node.moduleIndex() 聚类的编号
#             idx2label[node.physicalId] = node.moduleIndex()
#             if node.moduleIndex() not in label2idx:
#                 label2idx[node.moduleIndex()] = []
#             label2idx[node.moduleIndex()].append(node.physicalId)
#
#         node_count = 0
#         for k, v in label2idx.items():
#             if k == 0:
#                 node_count += len(v[2:])
#                 label2idx[k] = v[2:]
#                 # print(k, v[2:])
#             else:
#                 node_count += len(v[1:])
#                 label2idx[k] = v[1:]
#                 # print(k, v[1:])
#
#         # print(node_count)
#         # 孤立点个数
#         print("孤立点数：{}".format(len(single)))
#
#         keys_len = len(list(label2idx.keys()))
#         # print(keys_len)
#
#         # 孤立点放入到结果中
#         for single_node in single:
#             idx2label[single_node] = keys_len
#             label2idx[keys_len] = [single_node]
#             keys_len += 1
#
#         print("总类别数：{}".format(keys_len))
#
#         idx_len = len(list(idx2label.keys()))
#         print("总节点数：{}".format(idx_len))
#
#         # 保存结果
#         if save_result:
#             with open(pred_label_path, 'w') as of:
#                 for idx in range(idx_len):
#                     of.write(str(idx2label[idx]) + '\n')
#
#         if label_path is not None:
#             pred_labels = intdict2ndarray(idx2label)
#             true_lb2idxs, true_idx2lb = read_meta(label_path)
#             gt_labels = intdict2ndarray(true_idx2lb)
#             for metric in metrics:
#                 evaluate(gt_labels, pred_labels, metric)

# ------------------------------------- GAT

# 真正的 图注意力网络 2025年4月3日15:48:56 yzx
class GAT_Model(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.

    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.

    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.

    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )
        # self.trans = nn.Parameter(torch.ones(7, 1433), requires_grad=True)

    # dataset is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        # dataset[0]: torch.Size([2708, 1433])
        # dataset[1]: torch.Size([2, 13264])

        # self.gat_net(dataset)[0]: torch.Size([2708, 7])
        # self.gat_net(dataset)[1]: torch.Size([2, 13264])

        # 注意 dataset[1] == self.gat_net(dataset)[1]
        return self.gat_net(data)



class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.

    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        if layer_type == LayerType.IMP1:
            # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            # 可用于线性变化的W矩阵
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scores_source = nn.Linear(num_of_heads, 1, bias=True)
        self.scores_target = nn.Linear(num_of_heads, 1, bias=True)
        self.silu = nn.SiLU()
        self.elu = nn.GELU()

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight, gain=0.9)
        nn.init.xavier_uniform_(self.scoring_fn_target, gain=0.9)
        nn.init.xavier_uniform_(self.scoring_fn_source, gain=0.9)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


# 实际使用的 2025年4月3日15:50:21 yzx
class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)
        # self.soft_weight = nn.Parameter(torch.FloatTensor(2708, 1, 7))
        # self.soft = nn.Softmax(dim=-1)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index = data  # unpack dataset
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'具有两种边  (2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        # 随机丢掉一部分样本的特征
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        # 1433 到 64 对 丢掉之后1433个特征 进行 全连接   64  拉伸  8 * 8
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        # nodes_features_proj = self.softmax(self.soft_weight * nodes_features_proj)
        # 随机丢掉拉伸后  8 * 8的特征
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
       # 随机给2 个初始化 即得分 ,  scores_source  scores_target
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_source_lifted = self.scores_source(scores_source_lifted)
        scores_target_lifted = self.scores_target(scores_target_lifted)
        scores_source_lifted = self.silu(scores_source_lifted)
        scores_target_lifted = self.elu(scores_target_lifted)
        # 通过两个全连接 和两个边 构造一个分数
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)

# torch.Size([2708, 64])       torch.Size([2, 13264])
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        # trg_index_broadcasted 只是拓展成八个
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        # 初始化 0 矩阵
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        # 成功 把 多个边的信息 加到 节点上面
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        # 再次反转成 边的形式
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class GATLayerImp2(GATLayer):
    """
        Implementation #2 was inspired by the official GAT implementation: https://github.com/PetarV-/GAT

        It's conceptually simpler than implementation #3 but computationally much less efficient.

        Note: this is the naive implementation not the sparse one and it's only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.

    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP2, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #

        in_nodes_features, connectivity_mask = data  # unpack dataset
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation (using sum instead of bmm + additional permute calls - compared to imp1)
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)
        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        #
        # Step 3: Neighborhood aggregation (same as in imp1)
        #

        # batch matrix multiply, shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        # Note: watch out here I made a silly mistake of using reshape instead of permute thinking it will
        # end up doing the same thing, but it didn't! The acc on Cora didn't go above 52%! (compared to reported ~82%)
        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        #
        # Step 4: Residual/skip connections, concat and bias (same as in imp1)
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)


class GATLayerImp1(GATLayer):
    """
        This implementation is only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.

    """
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP1, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, connectivity_mask = data  # unpack dataset
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (1, N, FIN) * (NH, FIN, FOUT) -> (NH, N, FOUT) where NH - number of heads, FOUT num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0), self.proj_param)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # batch matrix multiply, shape = (NH, N, FOUT) * (NH, FOUT, 1) -> (NH, N, 1)
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1, 2))
        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        #
        # Step 3: Neighborhood aggregation
        #

        # shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj)

        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.transpose(0, 1)

        #
        # Step 4: Residual/skip connections, concat and bias (same across all the implementations)
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)




#
# Helper functions
#
def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')


  #todo  所有模型的入口  yzx 2025年3月26日12:47:38
class ClusteringModel(nn.Module):
    def __init__(self, m, s, nclass=7, input_dim=1433, hdim=7, **kwargs):
        super(ClusteringModel, self).__init__()
        self.train_flag = True
        self.total_sample = 2708  # 0开始
        # 模型1
        # self.model = ssp_GCN_Model(input_dim=input_dim, out_dim=hdim)
        # 模型2
        self.model1 = GAT_Model(
            num_of_layers=kwargs['num_of_layers'],
            num_heads_per_layer=kwargs['num_heads_per_layer'],
            num_features_per_layer=kwargs['num_features_per_layer'],
            add_skip_connection=kwargs['add_skip_connection'],
            bias=kwargs['bias'],
            dropout=kwargs['dropout'],
            layer_type=kwargs['layer_type'],
            log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
        )

        num_features_per_layer = [hdim] + (kwargs['num_of_layers'] - 1) * [8] + [hdim]
        self.model2 = GAT_Model(
            num_of_layers=kwargs['num_of_layers'],
            num_heads_per_layer=kwargs['num_heads_per_layer'],
            num_features_per_layer=num_features_per_layer,
            add_skip_connection=kwargs['add_skip_connection'],
            bias=kwargs['bias'],
            dropout=kwargs['dropout'],
            layer_type=kwargs['layer_type'],
            log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
        )
        self.lam = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # 损失函数
        # self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        # self.loss_fn = AAMsoftmax(n_class=nclass, hdim=hdim, m=m, s=s)
        self.loss_fn = Control_Contrastive(n_class=nclass, hdim=hdim, m=m, s=s)
        self.expand = nn.Conv1d(1, nclass, 1)
        # self.loss_fn = KNN_Loss(init_w=1., init_b=0., hdim=hdim, n_class=nclass)
        #  优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=kwargs['lr'], weight_decay = kwargs['weight_decay'])
        # 调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=0.97)
        # self.scheduler  = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-8, max_lr=1e-3, step_size_up=120000//2, mode="triangular2", cycle_momentum=False)

        sys.stderr.write("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n")
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + " GPT_IConv_DGCN  para number = %.4fMB\n"%(sum(param.numel() for param in self.model1.parameters()) / 1024 / 1024))
        # print(time.strftime("%m-%d %H:%M:%S") + " Control_Contrastive_loss  para number = %.4fMB"%(sum(param.numel() for param in self.Control_Contrastive_loss.parameters()) / 1024 / 1024))
        # print(time.strftime("%m-%d %H:%M:%S") + " WavLMForXVector  para number = %.4fMB"%(sum(param.numel() for param in self.WavLMForXVector.parameters()) / 1024 / 1024))
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + " Total_Model  para number = %.4fMB\n"%(sum(param.numel() for param in self.parameters()) / 1024 / 1024))
        # 计算模型大小 和 参数计算
        # self.macs_params(input_size=(1, self.in_dim, 202), device="cuda:0")
        sys.stderr.write("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n\n")

    def forward(self, x, label, index_label, node_dim=0):
        # gcn
        # x = self.model(x[0], x[1]).index_select(node_dim, label)

        #gat
        x_res, e = self.model1(x)
        x, e = self.model2((x_res, e))
        lam = torch.clamp_(self.lam.clone(), min=0.3+1e-5, max=1 - 0.3 - 1e-5)
        x = lam * x_res + (1-lam) * x
        print(self.lam)

        # A, D = self.trans_adj(e, 0, self.total_sample-1)
        # loss = self.loss_fn(x, index_label)
        # self-supervise
        x = x.index_select(node_dim, label)
        # A = A.index_select(0, label).index_select(1, label)
        # D = D.index_select(0, label).index_select(1, label)
        # D = torch.diag(torch.sum(A, dim=0))   # 先保留这种方式
        # loss = self.loss_fn(self.expand(x.unsqueeze(-2)), A, D)

        # loss = self.loss_fn(x, A, D)

        loss = self.loss_fn(x, index_label)
        return x, loss if not None else 0

    def save_parameters(self, config, path):
        torch.save(get_training_state(config, self), path)
        # torch.save(self.state_dict(), path)

    def load_parameters(self, path, device):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=device, weights_only=True)['state_dict']
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                # name = name.replace("speaker_AAMsoftmax_loss", "Control_Contrastive_loss")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
            print("加载训练好的模型的参数部分：\n 名称 %s, 参数形状 %s ." % (origname, param.shape))

    def trans_adj(self, edge, min_value, max_value):
        need_len = max_value - min_value + 1
        min_index = min(torch.nonzero(edge[1] == min_value))
        max_index = max(torch.nonzero(edge[1] == max_value))
        A = torch.zeros((need_len, need_len), dtype=torch.float32, device=edge.device)
        for i in range(min_index, max_index+1):
            A[edge[0][i], edge[1][i]] += 1
        return A, torch.diag(torch.sum(A, dim=0))
    def laplacian(self, adjacencyMatrix, normalize=False):
        adj = torch.zeros((torch.max(adjacencyMatrix)+1, torch.max(adjacencyMatrix)+1), dtype=torch.int8, device=adjacencyMatrix.device)

        # 填充邻接矩阵
        adj[adjacencyMatrix[0], adjacencyMatrix[1]] = 1

        # 如果是无向图，需要对称化
        adjacencyMatrix = adj | adj.T  # 或 adj = adj + adj.T > 0

        numRows, numCols = adjacencyMatrix.shape
        degrees = torch.sum(adjacencyMatrix, axis=0)

        if normalize:
            normalizedDegrees = 1 / torch.sqrt( degrees.unsqueeze(0).T * adjacencyMatrix * degrees.unsqueeze(0))

            # remove inf's created by dividing by zero
            normalizedDegrees[normalizedDegrees == torch.inf] = 0

            return torch.diag(torch.ones(numRows, device=adjacencyMatrix.device)) - normalizedDegrees
        else:
            combinatorialLaplacian = torch.diag(degrees) - adjacencyMatrix
            return combinatorialLaplacian

    def train_network(self, dataLoader, args, phase, loop=1, preconditioner=None, lam=0.):
        labels = {}
        if phase == DataMode.TRAIN:
            self.train()
        else:
            self.eval()
        node_dim = 0  # node axis

        labels['train_index'] = dataLoader['node_labels'].index_select(node_dim, dataLoader['train_index'])
        labels['val_index'] = dataLoader['node_labels'].index_select(node_dim, dataLoader['val_index'])
        labels['test_index'] = dataLoader['node_labels'].index_select(node_dim, dataLoader['test_index'])

        # lr = self.optim.param_groups[0]['lr']
        # node_features shape = (N, FIN), edge_index shape = (2, E)
        graph_data = (
            dataLoader['node_features'], dataLoader['topology'])  # I pack dataset into tuples because GAT uses nn.Sequential which requires it
        for epoch in range(1, args.max_epoch+1):
            # self.scheduler.step(epoch - 1)  # StepLR 位置
            self.zero_grad()
            # Training loop
            train_scores, loss = self.forward(graph_data, dataLoader[self.get_node_index(phase)], labels[self.get_node_index(phase)], node_dim)
            # Validation loop
            with torch.no_grad():
                try:
                    valid_scores, val_loss = self.forward(graph_data, dataLoader['val_index'], labels['val_index'], node_dim)
                except Exception as e:  # "patience has run out" exception :O
                    print(str(e))
                    break  # break out from the training loop

            # 训练      主要的性能指标 ACC
            if phase == DataMode.TRAIN:
                self.optim.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                # if preconditioner:
                #     preconditioner.step(lam=lam)
                self.optim.step()  # apply the gradients to weights
                class_predictions = torch.argmax(train_scores, dim=-1)
                accuracy = 100 * torch.sum(torch.eq(class_predictions, labels['train_index']).long()).item() / len(labels['train_index'])
                metricsStr = f"acc : {accuracy:.2f}%"
                args.test_perf = -1
                tupleMetrics = accuracy,
            # 测试      主要的性能指标 ACC
            elif phase == DataMode.TEST:
                class_predictions = torch.argmax(train_scores, dim=-1)
                accuracy = 100 * torch.sum(torch.eq(class_predictions, labels['test_index']).long()).item() / len(labels['test_index'])

# NMI
                NMI = round(100 * metrics.normalized_mutual_info_score(labels['test_index'].detach().cpu(), class_predictions.detach().cpu()),
                            2)
           # ARI
                ARI = round(100 * metrics.adjusted_rand_score(labels['test_index'].detach().cpu(), class_predictions.detach().cpu()),
                            2)
                purity = round(100 * self.purity_score(labels['test_index'].detach().cpu(), class_predictions.detach().cpu())
                            ,2)
                metricsStr = f"acc : {accuracy:.2f}%, NMI : {NMI:.2f}%, ARI : {ARI:.2f}%, purity : {purity:.2f}%"
                tupleMetrics = accuracy, NMI, ARI, purity
                args.test_perf = accuracy
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             "hdim:%s,  %s, running loop [%5d] at running epoch [%5d], Loss: %.4f, LR: %.8f, metrics: %s \n" % (
                                 args.hdim, phase.name, loop, (loop-1) * args.max_epoch + epoch, loss, self.optim.param_groups[0]['lr'], metricsStr))
            # print(time.strftime("%m-%d %H:%M:%S") + \
            #       " %s, epoch [%5d], Loss: %.4f, LR: %.8f, ACC: %.2f%%" % (
            #           phase.name, epoch, loss, self.optim.param_groups[0]['lr'], accuracy))

            # 验证集部分 同样采用ACC
            val_class_predictions = torch.argmax(valid_scores, dim=-1)
            val_accuracy = 100 * torch.sum(torch.eq(val_class_predictions, labels['val_index']).long()).item() / len(labels['val_index'])
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " Validation, running epoch [%5d], Loss: %.4f, LR: %.8f, ACC: %.2f%%\n\n" % (
                                 (loop-1) * args.max_epoch + epoch, val_loss, self.optim.param_groups[0]['lr'], val_accuracy))
            # print(time.strftime("%m-%d %H:%M:%S") + \
            #       " VAL, epoch [%5d], Loss: %.4f, LR: %.8f, ACC: %.2f%% \n" % (
            #           epoch, val_loss, self.optim.param_groups[0]['lr'], val_accuracy))
            sys.stderr.flush()

        return loss, self.optim.param_groups[0]['lr'], tupleMetrics, args.max_epoch

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def get_node_index(self, phase):
        if phase == DataMode.TRAIN:
            return 'train_index'
        elif phase == DataMode.VAL:
            return 'val_index'
        else:
            return 'test_index'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj