import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
def fast_fft(x):
    return torch.fft.fft(x, dim=-1)
class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # 获取时间戳的时分年月日
        # 1000ms = 1s , 60s = 1min , 60 min = 60 * 60 * 1000ms = 3600000ms = 1h ,
        # 24h = 24 * 60 * 60 * 1000ms = 86400000ms = 1d , 365d = 36
        minute = x%60000
        hour = (x//60000)%24
        weekday = (x//86400000)%7
        day = (x//86400000)%31
        month = (x//2629800000)%12
        minute_x = self.minute_embed(minute) if hasattr(self, 'minute_embed') else 0
        hour_x = self.hour_embed(hour)
        weekday_x = self.weekday_embed(weekday)
        day_x = self.day_embed(day)
        month_x = self.month_embed(month)

        

        return hour_x + weekday_x + day_x + month_x + minute_x
class CommonMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layers=[]):
        super(CommonMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_layers = torch.nn.ModuleList()
        assert input_dim == layers[0] 
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(
                nn.Linear(in_size, out_size)
            )
        self.output_layer = nn.Linear(layers[-1], output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        vector = x
        for layer in self.fc_layers:
            vector = layer(vector)
        vector = self.output_layer(vector)
        return self.sigmoid(vector)
class FastKAN(torch.nn.Module):
    def __init__(
        self,
        config,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.Tanh, # 这个是：base_activation=torch.nn.SiLU ， SiLU是一个激活函数，可以调整
        grid_eps=0.02,  # 这个是：grid_eps=0.02 ， 0.02是一个超参数，可以调整，作用是：Kan的grid更新的时候，会有一个eps的扰动，这个扰动是为了防止grid更新的时候，出现过拟合的情况
        grid_range=[-1, 1],
    ):
        super(FastKAN, self).__init__()
        self.config = config
        self.num_users = config['num_users']  # 用户数
        self.num_items = config['num_items']  # 物品数
        self.latent_dim = config['latent_dim']  # 潜在维度
        layers_hidden = config['layers']
        self.grid_size = config['grid_size']
        grid_size = self.grid_size
        self.spline_order = spline_order

        # self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        # 用户嵌入
        #   
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # self.embedding_time = TemporalEmbedding(d_model=self.latent_dim)
        self.embed_dim = self.config['embed_dim']
        self.embedding_user = torch.nn.Linear(
            in_features=self.embed_dim, out_features=self.latent_dim) if config['use_jointembedding'] else torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        if config['use_transfermer']:
            self.multheadAttention_layer = TransformerBlockKan(
                input_dim=self.latent_dim,
                output_dim=self.latent_dim,
                use_kan=config['use_kan'],
                use_cuda=config['use_cuda'],
                use_mps=config['use_mps'],
            ) 
        input_dim = self.latent_dim 
        self.user_mlp = CommonMLP(
            input_dim=input_dim,
            output_dim=self.latent_dim,
            layers=[
                input_dim ,self.latent_dim*2
            ],
        )

       
        # self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.fc_layers = torch.nn.ModuleList()
        layers = config['layers']
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        # 线性输出层，输出层
        if config['use_cuda']:
            # 将模型转移到 GPU 上
            self.to(torch.device('cuda'))
            print("Model moved to GPU")
        elif config['use_mps']:
            self.to(torch.device('mps'))
            self.embedding_item.to(torch.device('mps'))
            self.embedding_user.to(torch.device('mps'))
            self.user_mlp.to(torch.device('mps'))
            self.multheadAttention_layer.to(torch.device('mps'))
            self.affine_output.to(torch.device('mps'))
            self.fc_layers.to(torch.device('mps'))
            print("Model moved to MPS")


    def forward(self, x  : torch.Tensor, uef : torch.Tensor = None):
        
        # 用户嵌入，这一步的操作是为了保持用户嵌入的维度与物品嵌入的维度一致，目的是为了后续的拼接操作
        items = x[0]
        
        
        if not self.config['use_jointembedding']:
            uef = torch.LongTensor([0 for i in range(len(items))])
        else:
            # user_embedding 重复 len(items) 次
            uef = uef.to(torch.float32)
        if self.config['use_cuda']:
            uef = uef.to(torch.device('cuda'))
        elif self.config['use_mps']:
            uef = uef.to(torch.device('mps'))
        user_embedding = None
        if self.config['use_jointembedding']:
            uef = self.embedding_user(uef)
            user_embedding = uef.repeat(len(items), 1)
        else:
            user_embedding = self.embedding_user(
                uef
            )
        

        # item_time = x[1]
        
        if self.config['use_cuda']:
            items = items.to(torch.device('cuda'))
            user_embedding = user_embedding.to(torch.device('cuda'))
            # item_time = item_time.to(torch.device('cuda'))
        elif self.config['use_mps']:
            items = items.to(torch.device('mps'))
            user_embedding = user_embedding.to(torch.device('mps'))
            # item_time = item_time.to(torch.device('cpu'))

        item_embedding = self.embedding_item(items)  # 物品嵌入
        item_embedding = item_embedding.to(torch.float32)

        # user_embedding = self.embedding_user(users)
        # time_embedding = self.embedding_time(item_time)
        if self.config['use_transfermer']:
            
            item_embedding = self.multheadAttention_layer(
                item_embedding,item_embedding,item_embedding,mask=None) 
        user_embedding = self.user_mlp(user_embedding)

        vector = torch.cat([ user_embedding , item_embedding], dim=-1)  # the concat latent vector
        for layer in self.fc_layers:
            vector = layer(vector)
        vector = self.affine_output(vector)  # output layer
        return self.sigmoid(vector)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.fc_layers
        )
import mlp as mlp
class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self,
        hid_dim,
        num_heads,
     dropout = 0.1,use_cuda=False,use_mps=False):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = num_heads


        # 强制 hid_dim 必须整除 h
        assert self.hid_dim % self.n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear( self.hid_dim, self.hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear( self.hid_dim, self.hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear( self.hid_dim, self.hid_dim)
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(torch.device('cuda')) if use_cuda else torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads]))
        if use_cuda:
            self.scale = self.scale.to(torch.device('cuda'))
        if use_mps:
            self.scale = self.scale.to(torch.device('mps'))

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class TransformerBlockKan(nn.Module):
    def __init__(self, 
        input_dim,
        output_dim, dropout=0.1, use_kan=True,use_cuda=False,use_mps=False):
        super(TransformerBlockKan, self).__init__()
        self.attention = MultiheadAttention(hid_dim=input_dim, num_heads=8,use_cuda=use_cuda,use_mps=use_mps)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = None
        if use_kan == True:
            self.feed_forward = nn.Sequential(
                FastKANLayer(
                        input_dim,
                        input_dim * 2,
                        grid_max=3,
                        grid_min=-3,
                        base_activation = F.tanh,
                    ),
                    # FastKANLayer(embed_size, forward_expansion * embed_size),
                    # nn.ReLU(),
                FastKANLayer(
                        input_dim * 2,
                        output_dim,
                        grid_max=3,
                        grid_min=-3,
                        base_activation = F.tanh,
                    )  
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(input_dim, input_dim*2),
                nn.ReLU(),
                nn.Linear(input_dim*2, output_dim)
            )
        self.dropout = nn.Dropout(dropout)
        if use_cuda:
            self.to(torch.device('cuda'))
        if use_mps:
            self.to(torch.device('mps'))

    def forward(self, query, key, value, mask = None    ):
     
        attention = self.attention(query,key, value
                                   , mask)
       
        attention = attention.squeeze(1)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out


import engine as engine
import utils as utils
class FastKANEngine(engine.Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = FastKAN(config)
        if config['use_cuda'] is True:
            utils.use_cuda(True, config['device_id'])
            self.model
        elif config['use_mps']:
            device = utils.use_mps(True)
            config['device'] = device
            self.model = self.model.to(device)
        else:
            config['device'] = torch.device('cpu')
        super(FastKANEngine, self).__init__(config)
        print(self.model)

        
