from typing import Callable, Optional, Union,Tuple

import torch
from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_geometric.nn.inits import reset



class BA_GIN_Conv(MessagePassing):

    def __init__(self, nn: Callable,  in_channels: Union[int, Tuple[int, int]], out_channels: int, edge_feat: int, dropout: float = 0.0, negative_slope: float = 0.2, eps: float = 0.,train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.edge_feat = edge_feat
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.lin_e = Linear(edge_feat, out_channels, bias=False)
        self.a = torch.nn.Parameter(torch.zeros(size=(3 * out_channels, 1)))
        self.dropout = dropout
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.lin)
        reset(self.lin_e)
        self.eps.data.fill_(self.initial_eps)
        torch.nn.init.xavier_uniform_(self.a)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_attr: Optional[Tensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x) #(source_features, target_features)
        #edge_index, _ = add_remaining_self_loops(edge_index)
        # propagate_type: (x: OptPairTensor)
        h = self.lin(x[0]) #对邻居节点的向量进行线性变换
        e = self.lin_e(edge_attr)
        out = self.propagate(edge_index, x=h, size=size, edge_attr=e)

        x_r = x[1]#获取目标节点，也是中心节点的特征向量
        x_r = self.lin(x_r) #对中心节点也做了线性变换
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, edge_index_i) -> Tensor:
        e = torch.matmul((torch.cat([x_i, x_j, edge_attr], dim=-1)), self.a)
        e = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, edge_index_i)
        alpha = F.dropout(alpha, self.dropout, self.training)
        return x_j * alpha

    # def message_and_aggregate(self, adj_t: SparseTensor,
    #                           x: OptPairTensor) -> Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'