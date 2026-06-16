import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
import networkx as nx
from einops import rearrange, repeat

# 注意：Main.py 会先设置 CUDA_VISIBLE_DEVICES，再 import 本文件。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(0)
            dots = dots + attn_bias.to(dots.device, dtype=dots.dtype)

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Attention2(nn.Module):
    """CNN 光谱域注意力分支：返回 attention matrix。"""
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_bias=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(0)
            dots = dots + attn_bias.to(dots.device, dtype=dots.dtype)

        attn = dots.softmax(dim=-1)
        attn = attn.squeeze(1)
        return attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel, num_channel, [1, 2], 1, 0))

    def forward(self, x, mask=None, attn_bias=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask, attn_bias=attn_bias)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)
                    ).squeeze(3)
                x = attn(x, mask=mask, attn_bias=attn_bias)
                x = ff(x)
                nl += 1
        return x


class Transformer2(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Attention2(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.mode = mode

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return x


class ViT2(nn.Module):
    def __init__(self, patch_dim, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer2(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        x = x.to(torch.float32)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        pos = self.pos_embedding[:, :n]
        x = x + pos
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class SSConv(nn.Module):
    """Spectral-Spatial Convolution，保留原结构。"""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class MultiHeadBlockCNN(nn.Module):
    def __init__(self, num_heads, input_size):
        super(MultiHeadBlockCNN, self).__init__()
        if input_size % num_heads != 0:
            raise ValueError(
                "The CNN input size (%d) is not a multiple of the number of attention heads (%d)"
                % (input_size, num_heads)
            )
        self.num_heasds = num_heads
        self.head_size = int(input_size / num_heads)
        self.attn_conv = nn.ModuleList()

        for _ in range(num_heads):
            self.attn_conv.append(nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2))

    def cut_to_heads(self, x):
        b, c, h, w = x.shape
        x = x.reshape([b, self.num_heasds, self.head_size, h, w])
        return x.permute(1, 0, 2, 3, 4)

    def forward(self, input_tensor):
        multi_tensor = self.cut_to_heads(input_tensor)
        output_list = []
        for i in range(self.num_heasds):
            avg_out = torch.mean(multi_tensor[i], dim=1, keepdim=True)
            x = self.attn_conv[i](avg_out)
            out = torch.sigmoid(x).mul(multi_tensor[i])
            output_list.append(out)
        output_tensor = torch.cat(output_list, dim=1)
        return output_tensor


class CNNConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k // 2, stride=1, groups=ch_out)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out(x))
        return x


class SCGR(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,
                 Q, A, S, Edge_index, Edge_atter, SP_size: int, CNN_nhid,
                 Seg=None):
        super(SCGR, self).__init__()
        self.height = height
        self.width = width
        self.class_count = class_count
        self.channel = changel
        self.num_node = int(SP_size)

        if Seg is None and Q is None:
            raise ValueError("SCGR requires Seg for sparse superpixel mapping, or dense Q for fallback.")

        self.use_sparse_superpixel = Seg is not None
        if self.use_sparse_superpixel:
            seg_flatten = torch.as_tensor(Seg, dtype=torch.long, device=device).reshape(-1)
            if seg_flatten.numel() != height * width:
                raise ValueError(
                    f"Seg size mismatch: Seg has {seg_flatten.numel()} pixels, "
                    f"but image has {height * width} pixels."
                )
            if int(seg_flatten.max().item()) + 1 != self.num_node:
                print(
                    f"[Warning] SP_size={self.num_node}, but Seg max+1={int(seg_flatten.max().item()) + 1}. "
                    f"Use Seg max+1 as num_node."
                )
                self.num_node = int(seg_flatten.max().item()) + 1

            counts = torch.bincount(seg_flatten, minlength=self.num_node).float().clamp_min(1.0)
            self.register_buffer("seg_flatten", seg_flatten)
            self.register_buffer("seg_counts", counts.unsqueeze(1))
            self.Q = None
            self.norm_col_Q = None
        else:
            # 小图像兼容旧 dense Q。大图像不建议用这个分支。
            self.Q = Q
            self.norm_col_Q = Q / torch.sum(Q, 0, keepdim=True).clamp_min(1e-12)

        layers_count = 4

        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # Pixel-level Convolutional Sub-Network，保留但 forward 中主要使用多尺度 CNNlayerA/B/C。
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        # Superpixel Graph Transformer branch
        num_patches = self.num_node
        dim = 32
        depth = 5
        heads = 4
        mlp_dim = 8
        dropout = 0.1
        emb_dropout = 0.1
        dim_head = 16
        patch_dim = 128

        self.graph_dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, 'ViT')
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        graph_x = torch.as_tensor(S, dtype=torch.float32, device=device)
        graph_edge_index = Edge_index.to(device=device, dtype=torch.long)
        graph_edge_attr = Edge_atter.to(device=device, dtype=torch.long).view(-1).clamp(min=0, max=3)
        self.graph_data = data_process(graph_x, graph_edge_index, graph_edge_attr, None, self.num_node)
        self.graph_encoder = GraphormerEncoder(dim)

        # CNN Conv branch
        self.CNN_nhid = int(CNN_nhid)
        self.CNNlayerA1 = CNNConvBlock(self.channel, self.CNN_nhid, 7, self.height, self.width)
        self.CNNlayerA2 = CNNConvBlock(self.CNN_nhid, self.CNN_nhid, 7, self.height, self.width)
        self.CNNlayerA3 = CNNConvBlock(self.CNN_nhid, self.CNN_nhid, 7, self.height, self.width)

        self.CNNlayerB1 = CNNConvBlock(self.channel, self.CNN_nhid, 5, self.height, self.width)
        self.CNNlayerB2 = CNNConvBlock(self.CNN_nhid, self.CNN_nhid, 5, self.height, self.width)
        self.CNNlayerB3 = CNNConvBlock(self.CNN_nhid, self.CNN_nhid, 5, self.height, self.width)

        self.CNNlayerC1 = CNNConvBlock(self.channel, self.CNN_nhid, 3, self.height, self.width)
        self.CNNlayerC2 = CNNConvBlock(self.CNN_nhid, self.CNN_nhid, 3, self.height, self.width)
        self.CNNlayerC3 = CNNConvBlock(self.CNN_nhid, self.CNN_nhid, 3, self.height, self.width)

        CNN_nhead = 6
        self.CNN_hidden_size = 3 * self.CNN_nhid
        self.CNN_Multihead = MultiHeadBlockCNN(CNN_nhead, self.CNN_hidden_size)

        self.Tr_net = ViT2(
            patch_dim=1,
            num_patches=self.CNN_hidden_size,
            num_classes=64,
            dim=100,
            depth=1,
            heads=1,
            mlp_dim=8,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.fusion_norm = nn.LayerNorm(self.CNN_hidden_size + dim)
        self.Softmax_linear = nn.Sequential(nn.Linear(self.CNN_hidden_size + dim, self.class_count))
        self.cnn_spectral_gamma = nn.Parameter(torch.tensor(0.02))

    def _pixel_to_superpixel_mean(self, pixel_features):
        """
        pixel_features: [H*W, C]
        return: [num_superpixels, C]
        """
        sums = pixel_features.new_zeros((self.num_node, pixel_features.shape[1]))
        sums.index_add_(0, self.seg_flatten, pixel_features)
        return sums / self.seg_counts.to(pixel_features.device, dtype=pixel_features.dtype)

    def _superpixel_to_pixel(self, superpixel_features):
        """
        superpixel_features: [num_superpixels, C]
        return: [H*W, C]
        """
        return superpixel_features[self.seg_flatten]

    def forward(self, x: torch.Tensor, y_flatten):
        x_origin = x
        h, w, c = x.shape
        y_flatten = y_flatten.to(x.device, dtype=torch.long)

        # Spectra Transformation Sub-Network
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        clean_x = torch.squeeze(noise, 0).permute([1, 2, 0])

        clean_x_flatten = clean_x.reshape([h * w, -1])
        if self.use_sparse_superpixel:
            superpixels_flatten = self._pixel_to_superpixel_mean(clean_x_flatten)
        else:
            superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)

        # Superpixel Graph Transformer branch
        sp_x = superpixels_flatten.to(torch.float32)
        sp_x = self.patch_to_embedding(sp_x)
        n, d = sp_x.shape

        graph_data = self.graph_encoder(self.graph_data)
        degree_encoding = graph_data.degree_encoding.to(sp_x.device, dtype=sp_x.dtype)
        pos = self.pos_embedding[:n, :]
        sp_x = sp_x + pos + degree_encoding
        sp_x = self.dropout(sp_x)

        sp_x = sp_x.unsqueeze(0)
        attn_bias = graph_data.attn_bias.view(1, 4, n, n)
        sp_x = self.transformer(sp_x, mask=None, attn_bias=attn_bias)
        sp_x = self.mlp_head(sp_x).squeeze(0)

        if self.use_sparse_superpixel:
            transformer_pixel = self._superpixel_to_pixel(sp_x)
            Transformer_result = transformer_pixel[y_flatten]
        else:
            Transformer_result = torch.matmul(self.Q, sp_x)
            Transformer_result = Transformer_result[y_flatten]

        # CNN multi-scale branch
        CNNin = torch.unsqueeze(x_origin.permute([2, 0, 1]), 0)

        CNNmid1_A = self.CNNlayerA1(CNNin)
        CNNmid1_B = self.CNNlayerB1(CNNin)
        CNNmid1_C = self.CNNlayerC1(CNNin)
        CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C

        CNNmid2_A = self.CNNlayerA2(CNNin)
        CNNmid2_B = self.CNNlayerB2(CNNin)
        CNNmid2_C = self.CNNlayerC2(CNNin)
        CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C

        CNNout_A = self.CNNlayerA3(CNNin)
        CNNout_B = self.CNNlayerB3(CNNin)
        CNNout_C = self.CNNlayerC3(CNNin)

        CNNout = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1)
        CNNout = self.CNN_Multihead(CNNout)
        CNNout = torch.squeeze(CNNout, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        CNN_attention = CNNout[y_flatten].unsqueeze(2)
        attention_x = self.Tr_net(CNN_attention)
        spectral_out = torch.einsum('bij,bjd->bid', attention_x, CNN_attention).squeeze(2)

        gamma = torch.clamp(self.cnn_spectral_gamma, 0.0, 0.2)
        cnn_result = (1.0 - gamma) * CNN_attention.squeeze(2) + gamma * spectral_out

        Y1 = torch.cat([cnn_result, Transformer_result], dim=-1)
        Y1 = self.fusion_norm(Y1)
        Y = self.Softmax_linear(Y1)
        return Y


def data_process(S, Edge_index, Edge_atter, y, num_node):
    data = Data(x=S, edge_index=Edge_index, edge_attr=Edge_atter, y=y)
    data.num_nodes = num_node
    data.num_edges = Edge_index.shape[1]
    data.batch = torch.zeros(num_node, dtype=torch.int64, device=device)
    data = graphormer_pre_processing(data, 20)
    return data


def graphormer_pre_processing(data, distance):
    graph: nx.DiGraph = to_networkx(data)

    data.in_degrees = torch.tensor([d for _, d in graph.in_degree()], device=device)
    data.out_degrees = torch.tensor([d for _, d in graph.out_degree()], device=device)

    max_in_degree = torch.max(data.in_degrees)
    max_out_degree = torch.max(data.out_degrees)
    if max_in_degree >= 512:
        raise ValueError(f"Encountered in_degree: {max_in_degree}, increase num_in_degrees.")
    if max_out_degree >= 512:
        raise ValueError(f"Encountered out_degree: {max_out_degree}, increase num_out_degrees.")

    N = len(graph.nodes)
    shortest_paths = nx.shortest_path(graph)

    spatial_types = torch.empty(N ** 2, dtype=torch.long, device=device).fill_(distance)
    graph_index = torch.empty(2, N ** 2, dtype=torch.long, device=device)

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        shortest_path_types = torch.zeros(N ** 2, distance, dtype=torch.long, device=device)
        edge_attr = torch.zeros(N, N, dtype=torch.long, device=device)
        edge_attr[data.edge_index[0], data.edge_index[1]] = data.edge_attr

    for i in range(N):
        base = i * N
        graph_index[0, base:base + N] = i
        graph_index[1, base:base + N] = torch.arange(N, device=device)

    for i, paths in shortest_paths.items():
        for j, path in paths.items():
            if len(path) > distance:
                path = path[:distance]
            spatial_types[i * N + j] = len(path) - 1

            if len(path) > 1 and hasattr(data, "edge_attr") and data.edge_attr is not None:
                path_attr = [edge_attr[path[k], path[k + 1]] for k in range(len(path) - 1)]
                shortest_path_types[i * N + j, :len(path) - 1] = torch.tensor(path_attr, dtype=torch.long, device=device)

    data.spatial_types = spatial_types
    data.graph_index = graph_index
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.shortest_path_types = shortest_path_types
    return data


BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)


class BiasEncoder(torch.nn.Module):
    def __init__(self, num_heads: int, num_spatial_types: int,
                 num_edge_types: int, use_graph_token: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.spatial_encoder = torch.nn.Embedding(num_spatial_types + 1, num_heads).to(device)
        self.edge_dis_encoder = torch.nn.Embedding(num_spatial_types * num_heads * num_heads, 1).to(device)
        self.edge_encoder = torch.nn.Embedding(num_edge_types, num_heads).to(device)
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        self.edge_encoder.weight.data.normal_(std=0.02)
        self.edge_dis_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        spatial_types = self.spatial_encoder(data.spatial_types)
        spatial_encodings = to_dense_adj(data.graph_index, data.batch, spatial_types)
        bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)

        if hasattr(data, "shortest_path_types"):
            edge_types = self.edge_encoder(data.shortest_path_types)
            edge_encodings = to_dense_adj(data.graph_index, data.batch, edge_types)

            spatial_distances = to_dense_adj(data.graph_index, data.batch, data.spatial_types)
            spatial_distances = spatial_distances.float().clamp(min=1.0).unsqueeze(1)

            B, N, _, max_dist, H = edge_encodings.shape
            edge_encodings = edge_encodings.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_encodings = torch.bmm(
                edge_encodings,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)
            )
            edge_encodings = edge_encodings.reshape(max_dist, B, N, N, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_encodings = edge_encodings.sum(-2).permute(BATCH_HEAD_NODE_NODE) / spatial_distances
            bias += edge_encodings

        if self.use_graph_token:
            bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            bias[:, :, 1:, 0] = self.graph_token
            bias[:, :, 0, :] = self.graph_token

        B, H, N, _ = bias.shape
        data.attn_bias = bias.reshape(B * H, N, N)
        return data


class NodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_in_degree, num_out_degree,
                 input_dropout=0.0, use_graph_token: bool = True):
        super().__init__()
        self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim).to(device)
        self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim).to(device)

    def forward(self, data):
        in_degree_encoding = self.in_degree_encoder(data.in_degrees.data)
        out_degree_encoding = self.out_degree_encoder(data.out_degrees.data)
        data.degree_encoding = in_degree_encoding + out_degree_encoding
        return data


class GraphormerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        encoders = [
            BiasEncoder(
                4,
                20,
                4,
                False
            ),
            NodeEncoder(
                dim_emb,
                512,
                512,
                input_dropout=0.0,
            ),
        ]
        super().__init__(*encoders)


# ============================================================
# Multimodal extension: SCGR-aligned auxiliary-source branch +
# adaptive conservative HSI-guided auxiliary fusion.
#
# This version is tuned for the trend observed on Houston2013 and Augsburg:
#   1) The auxiliary branch keeps the original pixel token and same-Seg
#      superpixel token, and additionally adds a local-detail token
#      (pixel feature minus its superpixel mean). This is meant to improve AA
#      by preserving boundary/minority-class information instead of averaging it
#      away at the superpixel level.
#   2) The auxiliary input is decomposed into raw / smoothed / high-pass maps
#      before the multiscale CNN. LiDAR benefits from local height details;
#      SAR benefits from explicit smoothing and detail separation.
#   3) Cross-modal fusion remains conservative: HSI is still the main branch,
#      but the fixed global gates are now modulated by a sample-wise reliability
#      gate and an optional class-wise delta gate. Their initialization keeps the
#      initial behavior identical to the old global gates.
# ============================================================


class CMlp(nn.Module):
    """Small FFN used in the conservative fusion encoder."""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AuxFeatureCalibrator(nn.Module):
    """Lightweight channel calibration for auxiliary CNN features."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 8)
        self.norm = nn.BatchNorm2d(channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.norm(x)
        return x_norm * self.gate(x_norm)


class CLSCrossAttention(nn.Module):
    """
    CLS-token attention used by the fusion encoder.
    Only the first token is updated; the remaining tokens are memory tokens.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, use_channel=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"fusion_dim={dim} must be divisible by fusion_heads={num_heads}.")
        self.num_heads = num_heads
        self.use_channel = use_channel
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        if use_channel:
            hidden = max(dim // 4, 16)
            self.channel_attn = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(inplace=False),
                nn.Linear(hidden, dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, n, c = x.shape
        h = self.num_heads
        q = self.q(x[:, 0:1]).reshape(b, 1, h, c // h).transpose(1, 2)
        k = self.k(x).reshape(b, n, h, c // h).transpose(1, 2)
        v = self.v(x).reshape(b, n, h, c // h).transpose(1, 2)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2).reshape(b, 1, c)
        out = self.drop(self.proj(out))
        if self.use_channel:
            out = out * self.channel_attn(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, use_channel=True):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.attn = CLSCrossAttention(dim, num_heads=heads, dropout=dropout, use_channel=use_channel)
        self.ffn = CMlp(dim, hidden_dim=dim * 4, dropout=dropout)

    def forward(self, x):
        cls = x[:, 0:1] + self.attn(self.attn_norm(x))
        memory_tokens = x[:, 1:]
        updated_cls = cls + self.ffn(self.ffn_norm(cls))
        return torch.cat([updated_cls, memory_tokens], dim=1)


class FusionEncoder(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dropout=0.1, use_channel=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock(dim, heads=heads, dropout=dropout, use_channel=use_channel)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


class AuxSourceBranch(nn.Module):
    """
    SCGR-aligned auxiliary-source branch for LiDAR / SAR / DSM.

    Outputs up to three tokens:
      - aux_pixel_token      : local pixel-level auxiliary context;
      - aux_superpixel_token : same-Seg superpixel context;
      - aux_detail_token     : local detail relative to the HSI superpixel region.

    The third token is important for the two target cases:
      - Houston/LiDAR: preserves local elevation discontinuities around object borders;
      - Augsburg/SAR : separates local texture/detail from the smoothed region context.
    """
    def __init__(self, in_ch, aux_nhid, fusion_dim, height, width,
                 seg_flatten=None, seg_counts=None, aux_cnn_heads=3,
                 use_input_detail=True, use_sp_detail_token=True,
                 aux_profile='generic'):
        super().__init__()
        self.in_ch = int(in_ch)
        self.height = int(height)
        self.width = int(width)
        self.aux_nhid = int(aux_nhid)
        self.use_input_detail = bool(use_input_detail)
        self.use_sp_detail_token = bool(use_sp_detail_token)
        self.num_aux_tokens = 3 if self.use_sp_detail_token else 2
        self.aux_profile = str(aux_profile or 'generic').lower()
        component_init, detail_kernel = self._resolve_input_component_profile(self.aux_profile)
        self.detail_kernel = int(detail_kernel)

        stem_in_ch = self.in_ch * 3 if self.use_input_detail else self.in_ch
        self.input_stem = nn.Sequential(
            nn.BatchNorm2d(stem_in_ch),
            nn.Conv2d(stem_in_ch, self.in_ch, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        # Learnable raw / smoothed / high-pass component weights.
        # DSM is initialized to rely more on smoothed elevation and less on noisy high-pass detail;
        # LiDAR keeps stronger detail because object boundaries usually benefit from height discontinuities.
        component_init = torch.as_tensor(component_init, dtype=torch.float32).clamp_min(1e-4)
        self.input_component_logits = nn.Parameter(self._inverse_softplus(component_init))

        self.A1 = CNNConvBlock(self.in_ch, self.aux_nhid, 7, self.height, self.width)
        self.A2 = CNNConvBlock(self.aux_nhid, self.aux_nhid, 7, self.height, self.width)

        self.B1 = CNNConvBlock(self.in_ch, self.aux_nhid, 5, self.height, self.width)
        self.B2 = CNNConvBlock(self.aux_nhid, self.aux_nhid, 5, self.height, self.width)

        self.C1 = CNNConvBlock(self.in_ch, self.aux_nhid, 3, self.height, self.width)
        self.C2 = CNNConvBlock(self.aux_nhid, self.aux_nhid, 3, self.height, self.width)

        self.aux_hidden_size = 3 * self.aux_nhid
        self.aux_multihead = MultiHeadBlockCNN(aux_cnn_heads, self.aux_hidden_size)
        self.feature_calib = AuxFeatureCalibrator(self.aux_hidden_size)

        self.pixel_proj = nn.Sequential(
            nn.LayerNorm(self.aux_hidden_size),
            nn.Linear(self.aux_hidden_size, fusion_dim)
        )
        self.superpixel_proj = nn.Sequential(
            nn.LayerNorm(self.aux_hidden_size),
            nn.Linear(self.aux_hidden_size, fusion_dim)
        )
        self.detail_proj = nn.Sequential(
            nn.LayerNorm(self.aux_hidden_size),
            nn.Linear(self.aux_hidden_size, fusion_dim)
        )
        self.token_norm = nn.LayerNorm(fusion_dim)

        if seg_flatten is not None:
            seg_flatten = torch.as_tensor(seg_flatten, dtype=torch.long).reshape(-1)
            if seg_flatten.numel() != self.height * self.width:
                raise ValueError(
                    f"seg_flatten has {seg_flatten.numel()} pixels, but expected {self.height * self.width}."
                )
            self.register_buffer('aux_seg_flatten', seg_flatten)
            if seg_counts is None:
                num_node = int(seg_flatten.max().item()) + 1
                seg_counts = torch.bincount(seg_flatten, minlength=num_node).float().clamp_min(1.0).unsqueeze(1)
            self.register_buffer('aux_seg_counts', torch.as_tensor(seg_counts, dtype=torch.float32).reshape(-1, 1))
        else:
            self.aux_seg_flatten = None
            self.aux_seg_counts = None

    @staticmethod
    def _inverse_softplus(x):
        return torch.log(torch.expm1(x).clamp_min(1e-6))

    @staticmethod
    def _resolve_input_component_profile(aux_profile):
        profile = str(aux_profile or 'generic').lower()
        has_dsm = 'dsm' in profile or 'dem' in profile
        has_sar = 'sar' in profile
        has_lidar = 'lidar' in profile or 'li' == profile
        if has_dsm and not has_sar:
            # DSM on Augsburg can be helpful but is easy to overfit through local noise.
            # Use a larger smoothing window and a weak high-pass component.
            return (1.00, 1.25, 0.35), 5
        if has_dsm and has_sar:
            return (1.00, 1.15, 0.50), 5
        if has_sar:
            return (1.00, 1.15, 0.65), 3
        if has_lidar:
            return (1.00, 0.80, 1.00), 3
        return (1.00, 1.00, 0.75), 3

    def _one_to_bchw(self, aux, ref_device):
        aux = torch.as_tensor(aux, dtype=torch.float32, device=ref_device)
        if aux.dim() == 2:
            aux = aux.unsqueeze(0).unsqueeze(0)              # [1, 1, H, W]
        elif aux.dim() == 3:
            # CHW usually has a small first dimension; HWC usually has a small last dimension.
            if aux.shape[0] <= max(self.in_ch, 16) and aux.shape[1] > 16 and aux.shape[2] > 16:
                aux = aux.unsqueeze(0)                      # [C, H, W] -> [1, C, H, W]
            else:
                aux = aux.permute(2, 0, 1).unsqueeze(0)     # [H, W, C] -> [1, C, H, W]
        elif aux.dim() == 4:
            if aux.shape[-1] == self.in_ch and aux.shape[1] != self.in_ch:
                aux = aux.permute(0, 3, 1, 2)               # [B, H, W, C] -> [B, C, H, W]
        else:
            raise ValueError(f"Unsupported aux tensor shape: {tuple(aux.shape)}")

        if aux.shape[0] != 1:
            raise ValueError("This SCGR variant processes one full scene at a time; aux batch size must be 1.")
        if aux.shape[-2:] != (self.height, self.width):
            aux = F.interpolate(aux, size=(self.height, self.width), mode='bilinear', align_corners=False)
        return aux

    def _merge_aux(self, aux, ref_device):
        if isinstance(aux, dict):
            aux_list = list(aux.values())
        elif isinstance(aux, (list, tuple)):
            aux_list = list(aux)
        else:
            aux_list = [aux]
        aux = torch.cat([self._one_to_bchw(a, ref_device) for a in aux_list], dim=1)
        if aux.shape[1] != self.in_ch:
            raise ValueError(
                f"aux_changel={self.in_ch}, but received {aux.shape[1]} auxiliary channels after concatenation."
            )
        return aux

    def _pixel_to_superpixel_mean(self, pixel_features):
        """pixel_features: [H*W, C] -> [num_superpixels, C]"""
        if self.aux_seg_flatten is None:
            return None
        seg_flatten = self.aux_seg_flatten.to(pixel_features.device)
        seg_counts = self.aux_seg_counts.to(pixel_features.device, dtype=pixel_features.dtype)
        sums = pixel_features.new_zeros((seg_counts.shape[0], pixel_features.shape[1]))
        sums.index_add_(0, seg_flatten, pixel_features)
        return sums / seg_counts

    def _superpixel_to_pixel(self, superpixel_features):
        if self.aux_seg_flatten is None:
            return None
        return superpixel_features[self.aux_seg_flatten.to(superpixel_features.device)]

    def _prepare_aux_input(self, aux):
        if not self.use_input_detail:
            return self.input_stem(aux)
        k = int(self.detail_kernel)
        aux_smooth = F.avg_pool2d(aux, kernel_size=k, stride=1, padding=k // 2)
        aux_detail = aux - aux_smooth
        weights = F.softplus(self.input_component_logits).to(aux.device, dtype=aux.dtype)
        raw_w, smooth_w, detail_w = weights[0], weights[1], weights[2]
        aux_enhanced = torch.cat([raw_w * aux, smooth_w * aux_smooth, detail_w * aux_detail], dim=1)
        return self.input_stem(aux_enhanced)

    def forward(self, aux, y_flatten, ref_device):
        aux = self._merge_aux(aux, ref_device)
        aux = self._prepare_aux_input(aux)
        y_flatten = y_flatten.to(aux.device, dtype=torch.long)

        out_A = self.A2(self.A1(aux))
        out_B = self.B2(self.B1(aux))
        out_C = self.C2(self.C1(aux))

        out_cat = torch.cat([out_A, out_B, out_C], dim=1)
        out_cat = self.aux_multihead(out_cat)
        out_cat = self.feature_calib(out_cat)

        pixel_flat = torch.squeeze(out_cat, 0).permute(1, 2, 0).reshape(self.height * self.width, -1)
        aux_pixel_feat = pixel_flat[y_flatten]

        sp_feat = self._pixel_to_superpixel_mean(pixel_flat)
        if sp_feat is not None:
            sp_pixel = self._superpixel_to_pixel(sp_feat)
            aux_superpixel_feat = sp_pixel[y_flatten]
            aux_detail_feat = (pixel_flat - sp_pixel)[y_flatten]
        else:
            # Dense-Q fallback: keep the same token count by reusing pixel context.
            aux_superpixel_feat = aux_pixel_feat
            aux_detail_feat = aux_pixel_feat

        aux_pixel_token = self.pixel_proj(aux_pixel_feat)
        aux_superpixel_token = self.superpixel_proj(aux_superpixel_feat)
        token_list = [aux_pixel_token, aux_superpixel_token]
        if self.use_sp_detail_token:
            aux_detail_token = self.detail_proj(aux_detail_feat)
            token_list.append(aux_detail_token)
        aux_tokens = torch.stack(token_list, dim=1)
        return self.token_norm(aux_tokens)


class ConservativeHsiGuidedAuxFusion(nn.Module):
    """
    Conservative HSI-guided auxiliary fusion.

    The original SCGR logits are kept as the main prediction. This module only
    predicts a gated auxiliary residual logit. Compared with the previous version,
    the global gates are further modulated by sample-wise reliability and optional
    class-wise delta gating, both initialized to a neutral scale of 1.0.
    """
    def __init__(self, cnn_dim, graph_dim, class_count, fusion_dim=64,
                 aux_tokens=3, heads=4, depth=1, dropout=0.2,
                 use_cross_attention=True, use_gate=True,
                 use_channel=True, use_mbce=False,
                 aux_gate_init=-3.0, logit_gate_init=-3.0,
                 mbce_strength=0.05, nogate_residual_scale=0.1,
                 use_dynamic_gate=True, use_class_gate=True,
                 fusion_direction='hsi_to_aux',
                 use_agreement_gate=True, agreement_floor=0.50,
                 use_confidence_gate=True, confidence_floor=0.50):
        super().__init__()
        if fusion_dim % heads != 0:
            raise ValueError(f"fusion_dim={fusion_dim} must be divisible by fusion_heads={heads}.")
        direction = str(fusion_direction or 'hsi_to_aux').lower().replace('-', '_')
        direction_aliases = {
            'hsi2aux': 'hsi_to_aux',
            'hsi_to_auxiliary': 'hsi_to_aux',
            'oneway': 'hsi_to_aux',
            'one_way': 'hsi_to_aux',
            'single': 'hsi_to_aux',
            'single_way': 'hsi_to_aux',
            'aux2hsi': 'aux_to_hsi',
            'auxiliary_to_hsi': 'aux_to_hsi',
            'bi': 'bidirectional',
            'two_way': 'bidirectional',
            'none': 'none',
            'off': 'none',
        }
        direction = direction_aliases.get(direction, direction)
        if direction not in ['hsi_to_aux', 'aux_to_hsi', 'bidirectional', 'none']:
            raise ValueError(
                f"Unsupported fusion_direction={fusion_direction}. Use hsi_to_aux, aux_to_hsi, bidirectional, or none."
            )
        self.fusion_direction = direction
        self.use_cross_attention = bool(use_cross_attention) and direction != 'none'
        self.use_gate = bool(use_gate)
        self.use_mbce = bool(use_mbce)
        self.use_dynamic_gate = bool(use_dynamic_gate)
        self.use_class_gate = bool(use_class_gate)
        self.use_agreement_gate = bool(use_agreement_gate)
        self.use_confidence_gate = bool(use_confidence_gate)
        self.agreement_floor = float(agreement_floor)
        self.confidence_floor = float(confidence_floor)
        self.mbce_strength = float(mbce_strength)
        self.nogate_residual_scale = float(nogate_residual_scale)

        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.hsi_norm = nn.LayerNorm(fusion_dim)
        self.aux_norm = nn.LayerNorm(fusion_dim)

        self.hsi_reads_aux = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.aux_reads_hsi = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_drop = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(fusion_dim)
        self.aux_cross_norm = nn.LayerNorm(fusion_dim)

        if self.use_gate:
            self.aux_gate = nn.Parameter(torch.tensor(float(aux_gate_init)))
        else:
            self.register_buffer('aux_gate', torch.tensor(float(nogate_residual_scale)))

        # Neutral MBCE: tanh(0)=0, so initial scale is exactly 1.
        self.hsi_scale = nn.Parameter(torch.tensor(0.0))
        self.aux_scale = nn.Parameter(torch.tensor(0.0))

        gate_hidden = max(fusion_dim // 2, 16)
        self.dynamic_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        nn.init.zeros_(self.dynamic_gate[-1].weight)
        nn.init.zeros_(self.dynamic_gate[-1].bias)

        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.position_embeddings = nn.Parameter(torch.randn(1, 1 + 2 + aux_tokens, fusion_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder = FusionEncoder(fusion_dim, depth=depth, heads=heads,
                                     dropout=dropout, use_channel=use_channel)
        self.delta_classifier = nn.Linear(fusion_dim, class_count)
        nn.init.xavier_uniform_(self.delta_classifier.weight)
        nn.init.normal_(self.delta_classifier.bias, std=1e-6)

        self.class_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, class_count)
        )
        nn.init.zeros_(self.class_gate[-1].weight)
        nn.init.zeros_(self.class_gate[-1].bias)

        self.logit_gate = nn.Parameter(torch.tensor(float(logit_gate_init)))

    def _dynamic_reliability(self, hsi_tokens, aux_tokens):
        if not self.use_dynamic_gate:
            return hsi_tokens.new_ones((hsi_tokens.shape[0], 1))
        hsi_summary = hsi_tokens.mean(dim=1)
        aux_summary = aux_tokens.mean(dim=1)
        gate_input = torch.cat([
            hsi_summary,
            aux_summary,
            torch.abs(hsi_summary - aux_summary),
            hsi_summary * aux_summary
        ], dim=-1)
        # 2*sigmoid(0)=1.0, so the first forward pass keeps the old gate scale.
        return 2.0 * torch.sigmoid(self.dynamic_gate(gate_input))

    def _agreement_reliability(self, hsi_tokens, aux_tokens):
        if not self.use_agreement_gate:
            return hsi_tokens.new_ones((hsi_tokens.shape[0], 1))
        hsi_summary = hsi_tokens.mean(dim=1)
        aux_summary = aux_tokens.mean(dim=1)
        cosine = F.cosine_similarity(hsi_summary, aux_summary, dim=-1, eps=1e-6).unsqueeze(-1)
        floor = max(0.0, min(float(self.agreement_floor), 0.99))
        # cosine=0 keeps scale close to 1; positive agreement strengthens aux residual,
        # negative agreement suppresses it. This is especially useful for DSM-only Augsburg.
        return torch.clamp(1.0 + (1.0 - floor) * cosine, min=floor, max=2.0 - floor)

    def _confidence_reliability(self, base_logits, ref_tokens):
        if (not self.use_confidence_gate) or base_logits is None:
            return ref_tokens.new_ones((ref_tokens.shape[0], 1))
        with torch.no_grad():
            base_conf = torch.softmax(base_logits.detach(), dim=-1).max(dim=-1, keepdim=True).values
        floor = max(0.0, min(float(self.confidence_floor), 0.99))
        # Protect high-confidence HSI predictions while letting auxiliary residuals act on uncertain samples.
        return torch.clamp(floor + (1.0 - base_conf), min=floor, max=floor + 1.0)

    def forward(self, cnn_result, transformer_result, aux_tokens, base_logits=None):
        cnn_token = self.cnn_proj(cnn_result)
        graph_token = self.graph_proj(transformer_result)
        hsi_tokens = torch.stack([cnn_token, graph_token], dim=1)
        hsi_tokens = self.hsi_norm(hsi_tokens)
        aux_tokens = self.aux_norm(aux_tokens)

        reliability = (
            self._dynamic_reliability(hsi_tokens, aux_tokens)
            * self._agreement_reliability(hsi_tokens, aux_tokens)
            * self._confidence_reliability(base_logits, hsi_tokens)
        )

        if self.use_cross_attention:
            if self.use_gate:
                residual_scale = torch.sigmoid(self.aux_gate)
            else:
                residual_scale = self.aux_gate.to(hsi_tokens.device, dtype=hsi_tokens.dtype)

            if self.fusion_direction in ['hsi_to_aux', 'bidirectional']:
                # One-way default: HSI queries read auxiliary K/V.
                # Auxiliary information can correct HSI only through a gated residual.
                aux_delta, _ = self.hsi_reads_aux(hsi_tokens, aux_tokens, aux_tokens)
                aux_delta = self.cross_drop(self.cross_norm(aux_delta))
                hsi_tokens = hsi_tokens + residual_scale * reliability.view(-1, 1, 1) * aux_delta

            if self.fusion_direction in ['aux_to_hsi', 'bidirectional']:
                # Optional inverse direction for ablation only. The default preset keeps this disabled
                # because two-way feature overwriting was prone to negative transfer on Augsburg DSM.
                hsi_delta, _ = self.aux_reads_hsi(aux_tokens, hsi_tokens, hsi_tokens)
                hsi_delta = self.cross_drop(self.aux_cross_norm(hsi_delta))
                aux_tokens = aux_tokens + residual_scale * reliability.view(-1, 1, 1) * hsi_delta

        if self.use_mbce:
            hsi_tokens = (1.0 + self.mbce_strength * torch.tanh(self.hsi_scale)) * hsi_tokens
            aux_tokens = (1.0 + self.mbce_strength * torch.tanh(self.aux_scale)) * aux_tokens

        b = hsi_tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, hsi_tokens, aux_tokens], dim=1)
        tokens = tokens + self.position_embeddings[:, :tokens.shape[1], :]
        tokens = self.dropout(tokens)
        fused = self.encoder(tokens)
        delta_logits = self.delta_classifier(fused)

        if self.use_class_gate:
            # 2*sigmoid(0)=1.0, so this is neutral at initialization.
            class_scale = 2.0 * torch.sigmoid(self.class_gate(fused))
        else:
            class_scale = 1.0
        return torch.sigmoid(self.logit_gate) * reliability * class_scale * delta_logits


class SCGR_Multimodal(SCGR):
    """
    SCGR with a SCGR-aligned auxiliary branch and adaptive conservative HSI-guided fusion.

    Forward:
        logits = model(hsi, y_flatten, aux)

    If aux is None, or ablation_mode='exp1_hsi_only', the model falls back to
    the original HSI-only SCGR classifier.

    Ablation modes:
        exp1_hsi_only      : original SCGR, HSI-only fallback
        exp2_aux_concat    : auxiliary pixel/superpixel/detail tokens + CLS fusion, no cross attention
        exp3_cross_nogate  : HSI-guided cross attention, fixed small residual, no learned gate
        exp4_gated_cross   : HSI-guided gated cross attention, no MBCE
        exp5_full          : HSI-guided gated cross attention + weak neutral MBCE
    """
    def __init__(self, height: int, width: int, changel: int, class_count: int,
                 Q, A, S, Edge_index, Edge_atter, SP_size: int, CNN_nhid,
                 Seg=None, aux_changel: int = 1, aux_nhid=None,
                 fusion_dim: int = 64, fusion_heads: int = 4, fusion_depth: int = 1,
                 fusion_dropout: float = 0.2,
                 ablation_mode: str = None,
                 use_aux_branch: bool = True,
                 use_cross_attention: bool = True,
                 use_gate: bool = True,
                 use_channel: bool = True,
                 use_mbce: bool = False,
                 aux_gate_init: float = -3.0,
                 logit_gate_init: float = -3.0,
                 mbce_strength: float = 0.05,
                 nogate_residual_scale: float = 0.1,
                 aux_use_input_detail: bool = True,
                 aux_use_sp_detail_token: bool = True,
                 fusion_dynamic_gate: bool = True,
                 fusion_class_gate: bool = True,
                 aux_profile: str = 'generic',
                 fusion_direction: str = 'hsi_to_aux',
                 fusion_agreement_gate: bool = True,
                 fusion_confidence_gate: bool = True,
                 agreement_floor: float = 0.50,
                 confidence_floor: float = 0.50):
        super().__init__(height, width, changel, class_count,
                         Q, A, S, Edge_index, Edge_atter, SP_size, CNN_nhid, Seg=Seg)

        self.ablation_mode, use_aux_branch, use_cross_attention, use_gate, use_mbce = \
            self._resolve_ablation_switches(
                ablation_mode=ablation_mode,
                use_aux_branch=use_aux_branch,
                use_cross_attention=use_cross_attention,
                use_gate=use_gate,
                use_mbce=use_mbce
            )
        self.use_cross_attention = bool(use_cross_attention)
        self.use_gate = bool(use_gate)
        self.use_mbce = bool(use_mbce)

        self.aux_changel = int(aux_changel) if aux_changel is not None else 0
        self.use_aux_branch = bool(use_aux_branch) and self.aux_changel > 0
        if self.use_aux_branch:
            aux_nhid = int(aux_nhid) if aux_nhid is not None else max(16, int(CNN_nhid) // 2)
            seg_flatten = self.seg_flatten if hasattr(self, 'seg_flatten') else None
            seg_counts = self.seg_counts if hasattr(self, 'seg_counts') else None
            self.aux_branch = AuxSourceBranch(
                in_ch=self.aux_changel,
                aux_nhid=aux_nhid,
                fusion_dim=fusion_dim,
                height=height,
                width=width,
                seg_flatten=seg_flatten,
                seg_counts=seg_counts,
                aux_cnn_heads=3,
                use_input_detail=aux_use_input_detail,
                use_sp_detail_token=aux_use_sp_detail_token,
                aux_profile=aux_profile
            )
            self.cross_fusion = ConservativeHsiGuidedAuxFusion(
                cnn_dim=self.CNN_hidden_size,
                graph_dim=self.graph_dim,
                class_count=class_count,
                fusion_dim=fusion_dim,
                aux_tokens=self.aux_branch.num_aux_tokens,
                heads=fusion_heads,
                depth=fusion_depth,
                dropout=fusion_dropout,
                use_cross_attention=self.use_cross_attention,
                use_gate=self.use_gate,
                use_channel=use_channel,
                use_mbce=self.use_mbce,
                aux_gate_init=aux_gate_init,
                logit_gate_init=logit_gate_init,
                mbce_strength=mbce_strength,
                nogate_residual_scale=nogate_residual_scale,
                use_dynamic_gate=fusion_dynamic_gate,
                use_class_gate=fusion_class_gate,
                fusion_direction=fusion_direction,
                use_agreement_gate=fusion_agreement_gate,
                use_confidence_gate=fusion_confidence_gate,
                agreement_floor=agreement_floor,
                confidence_floor=confidence_floor
            )

    @staticmethod
    def _resolve_ablation_switches(ablation_mode=None, use_aux_branch=True,
                                   use_cross_attention=True, use_gate=True, use_mbce=False):
        if ablation_mode is None:
            return 'manual', bool(use_aux_branch), bool(use_cross_attention), bool(use_gate), bool(use_mbce)

        mode = str(ablation_mode).strip().lower().replace('-', '_')
        aliases = {
            'exp1': 'exp1_hsi_only',
            'hsi_only': 'exp1_hsi_only',
            'original': 'exp1_hsi_only',
            'ctfn': 'exp1_hsi_only',
            'exp2': 'exp2_aux_concat',
            'aux_concat': 'exp2_aux_concat',
            'concat': 'exp2_aux_concat',
            'exp3': 'exp3_cross_nogate',
            'cross_nogate': 'exp3_cross_nogate',
            'cross_no_gate': 'exp3_cross_nogate',
            'exp4': 'exp4_gated_cross',
            'gated_cross': 'exp4_gated_cross',
            'exp5': 'exp5_full',
            'full': 'exp5_full',
        }
        mode = aliases.get(mode, mode)

        if mode == 'exp1_hsi_only':
            return mode, False, False, False, False
        if mode == 'exp2_aux_concat':
            return mode, True, False, False, False
        if mode == 'exp3_cross_nogate':
            return mode, True, True, False, False
        if mode == 'exp4_gated_cross':
            return mode, True, True, True, False
        if mode == 'exp5_full':
            return mode, True, True, True, True
        raise ValueError(
            f"Unsupported ablation_mode={ablation_mode}. Use one of: "
            "exp1_hsi_only, exp2_aux_concat, exp3_cross_nogate, exp4_gated_cross, exp5_full."
        )

    def _extract_hsi_features(self, x: torch.Tensor, y_flatten):
        x_origin = x
        h, w, c = x.shape
        y_flatten = y_flatten.to(x.device, dtype=torch.long)

        # Spectra Transformation Sub-Network
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        clean_x = torch.squeeze(noise, 0).permute([1, 2, 0])

        clean_x_flatten = clean_x.reshape([h * w, -1])
        if self.use_sparse_superpixel:
            superpixels_flatten = self._pixel_to_superpixel_mean(clean_x_flatten)
        else:
            superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)

        # Superpixel Graph Transformer branch
        sp_x = superpixels_flatten.to(torch.float32)
        sp_x = self.patch_to_embedding(sp_x)
        n, d = sp_x.shape

        graph_data = self.graph_encoder(self.graph_data)
        degree_encoding = graph_data.degree_encoding.to(sp_x.device, dtype=sp_x.dtype)
        pos = self.pos_embedding[:n, :]
        sp_x = sp_x + pos + degree_encoding
        sp_x = self.dropout(sp_x)

        sp_x = sp_x.unsqueeze(0)
        attn_bias = graph_data.attn_bias.view(1, 4, n, n)
        sp_x = self.transformer(sp_x, mask=None, attn_bias=attn_bias)
        sp_x = self.mlp_head(sp_x).squeeze(0)

        if self.use_sparse_superpixel:
            transformer_pixel = self._superpixel_to_pixel(sp_x)
            transformer_result = transformer_pixel[y_flatten]
        else:
            transformer_result = torch.matmul(self.Q, sp_x)
            transformer_result = transformer_result[y_flatten]

        # Original SCGR multi-scale CNN branch
        CNNin = torch.unsqueeze(x_origin.permute([2, 0, 1]), 0)

        CNNmid1_A = self.CNNlayerA1(CNNin)
        CNNmid1_B = self.CNNlayerB1(CNNin)
        CNNmid1_C = self.CNNlayerC1(CNNin)
        CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C

        CNNmid2_A = self.CNNlayerA2(CNNin)
        CNNmid2_B = self.CNNlayerB2(CNNin)
        CNNmid2_C = self.CNNlayerC2(CNNin)
        CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C

        CNNout_A = self.CNNlayerA3(CNNin)
        CNNout_B = self.CNNlayerB3(CNNin)
        CNNout_C = self.CNNlayerC3(CNNin)

        CNNout = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1)
        CNNout = self.CNN_Multihead(CNNout)
        CNNout = torch.squeeze(CNNout, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        CNN_attention = CNNout[y_flatten].unsqueeze(2)
        attention_x = self.Tr_net(CNN_attention)
        spectral_out = torch.einsum('bij,bjd->bid', attention_x, CNN_attention).squeeze(2)

        gamma = torch.clamp(self.cnn_spectral_gamma, 0.0, 0.2)
        cnn_result = (1.0 - gamma) * CNN_attention.squeeze(2) + gamma * spectral_out
        return cnn_result, transformer_result

    def _base_logits(self, cnn_result, transformer_result):
        y1 = torch.cat([cnn_result, transformer_result], dim=-1)
        y1 = self.fusion_norm(y1)
        return self.Softmax_linear(y1)

    def forward(self, x: torch.Tensor, y_flatten, aux=None):
        y_flatten = y_flatten.to(x.device, dtype=torch.long)
        cnn_result, transformer_result = self._extract_hsi_features(x, y_flatten)
        base_logits = self._base_logits(cnn_result, transformer_result)

        if (not self.use_aux_branch) or aux is None:
            return base_logits

        aux_tokens = self.aux_branch(aux, y_flatten, ref_device=x.device)
        delta_logits = self.cross_fusion(cnn_result, transformer_result, aux_tokens, base_logits=base_logits)
        return base_logits + delta_logits


# Common aliases: use SCGR_Multimodal/SCGR_MM in Main.py.
# Legacy CTFN names are kept for compatibility with older scripts.
CTFN = SCGR
CTFN_Multimodal = SCGR_Multimodal
CTFN_MM = SCGR_Multimodal
SCGR_MM = SCGR_Multimodal
