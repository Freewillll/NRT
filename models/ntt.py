import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
import os 

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from datasets.dataset import *
from models.transformer import Transformer
from models.modules import ConvDropoutNonlinNorm, ConvDropoutNormNonlin


def pair3d(t):
    return t if isinstance(t, tuple) else (t, t, t)


def posemb_sincos_1d(seq, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *seq.shape, seq.device, seq.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)


def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, dim, d, h, w, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(d, device = device),
        torch.arange(w, device = device),
        torch.arange(h, device = device),
        indexing='ij'
    )

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=(3, 3, 3), stride=(1, 1, 1)):
        super(ResidualBlock, self).__init__()
        padding = tuple((k - 1) // 2 for k in kernel)
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(outchannel, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(outchannel, affine=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True),
                nn.InstanceNorm3d(outchannel, affine=True)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.leaky_relu(out, inplace=True)
        return out


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class NTT(nn.Module):
    def __init__(self, in_channels, base_num_filters, num_classes, down_kernel_list, stride_list,
                 dim, encoder_depth, decoder_depth, heads, mlp_dim, dropout, num_queries):
        super(NTT, self).__init__()
        assert len(down_kernel_list) == len(stride_list)
        self.downs = []

        # the first layer to process the input image
        self.pre_layer = nn.Sequential(
            ConvDropoutNormNonlin(in_channels, base_num_filters),
            ConvDropoutNormNonlin(base_num_filters, base_num_filters),
        )

        in_channels = base_num_filters
        out_channels = 2 * base_num_filters
        down_filters = []
        self.down_d = 1
        self.down_h = 1
        self.down_w = 1
        for i in range(len(down_kernel_list)):
            down_kernel = down_kernel_list[i]
            stride = stride_list[i]
            self.down_d *= stride[0]
            self.down_w *= stride[1]
            self.down_h *= stride[2]
            down_filters.append((in_channels, out_channels))
            down = ResidualBlock(in_channels, out_channels, kernel=down_kernel, stride=stride)
            self.downs.append(down)
            down = ResidualBlock(out_channels, out_channels, kernel=[3,3,3], stride=[1,1,1])
            self.downs.append(down)
            in_channels = out_channels
            out_channels = out_channels * 2

        out_channels = int(out_channels / 2)        
        
        self.input_proj = nn.Conv3d(out_channels, dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, dim)
        self.transformer = Transformer(dim, heads, encoder_depth, decoder_depth, mlp_dim, dropout)

        # convert layers to nn containers
        self.downs = nn.ModuleList(self.downs)
        self.class_head = nn.Linear(dim, num_classes)
        self.pos_head = MLP(dim, dim, 3, 3)

    def forward(self, img):
        assert img.ndim == 5
        img = self.pre_layer(img)
        ndown = len(self.downs)
        for i in range(ndown):
            img = self.downs[i](img)
        img = self.input_proj(img)
        pos = posemb_sincos_3d(img)
        hs = self.transformer(src=img, query_embed=self.query_embed.weight, pos_embed=pos)[0][0]
        
        output_class = self.class_head(hs)
        output_pos = self.pos_head(hs).sigmoid()
        out = {'pred_logits': output_class, 'pred_poses': output_pos}
        return out


if __name__ == '__main__':
    from torchinfo import summary
    import json
    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print('Initialize model...')

    img = torch.randn(2, 1, 64, 128, 128)
    model = NTT(**configs)
    print(model)
    outputs = model(img)
    # print(outputs)

    summary(model, input_size=[2, 1, 64, 128, 128])
