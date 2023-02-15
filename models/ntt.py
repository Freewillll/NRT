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
    _, d, w, h, dim, device, dtype = *patches.shape, patches.device, patches.dtype

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


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    # b, n
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    #  b, n
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)  # upper triangle
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=(3, 3, 3), stride=(1, 1, 1)):
        super(ResidualBlock, self).__init__()
        padding = tuple((k - 1) // 2 for k in kernel)
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(outchannel, affine=True),
            nn.ELU(alpha=0.2, inplace=True),
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
        out = F.elu(out, alpha=0.2, inplace=True)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, memory=None, attn_mask=None):
        x = self.norm(x)

        if memory is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        else:
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            kv = self.to_kv(memory).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        #  q, k, v dim:  b, h, n, d
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #  dots dim:  b, h, n, n
        if attn_mask is not None:
            assert attn_mask.size() == dots.size()
            dots.masked_fill_(attn_mask, float("-inf"))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Encoderlayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Decoderlayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x, memory, attn_mask):
        for self_attn, enc_attn, ff in self.layers:
            x = self_attn(x, attn_mask=attn_mask) + x
            x = enc_attn(x, memory=memory) + x
            x = ff(x) + x
        return x


class Encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_depth, image_width, image_height = pair3d(image_size)
        patch_depth, patch_width, patch_height = pair3d(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0,\
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_depth * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (w p2) (h p3) -> b d w h (p1 p2 p3 c)', p1=patch_depth, p2=patch_width, p3=patch_height),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.layers = Encoderlayer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, *_ = x.shape
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x = self.layers(x)
        x = x.mean(dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim)
        )
        self.heads = heads
        self.layers = Decoderlayer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, x, memory):
        # shape of x:   b, seq_len, seq_item_len, vec_len
        # x : start, item1, item2, ...
        mask_input = x[:, :, 0, -1] > 0
        attn_pad_mask = get_attn_pad_mask(mask_input, mask_input)
        attn_subsequent_mask = get_attn_subsequent_mask(mask_input)

        attn_mask = torch.gt((attn_pad_mask + attn_subsequent_mask), 0)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)   # b, h, n, n

        # -> b, n, d
        x = rearrange(x, 'b n ...  -> b n (...)')
        # b, dim  ->  b, n, dim
        memory = memory.unsqueeze(1).repeat(1, x.shape[1], 1) 
        # b, n, d
        x = self.embedding(x)
        pe = posemb_sincos_1d(x)
        x += pe
        x = self.layers(x, memory, attn_mask)
        return x


class NTT(nn.Module):
    def __init__(self, in_channels, base_num_filters, num_nodes, pos_dims, num_classes, down_kernel_list, stride_list, patch_size,
                 dim, depth, heads, dim_head, mlp_dim, img_shape):
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
            in_channels = out_channels
            out_channels = out_channels * 2

        out_channels = int(out_channels / 2)
        *_, d, w, h = img_shape
        d = int(d / self.down_d)
        h = int(h / self.down_h)
        w = int(w / self.down_w)

        self.encoder = Encoder(
            image_size=(d, w, h),
            patch_size=patch_size,
            channels=out_channels,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )

        self.decoder = Decoder(
            input_dim=num_nodes * (pos_dims + 1),
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )
        # convert layers to nn containers
        self.downs = nn.ModuleList(self.downs)
        self.proj = nn.Linear(dim, num_nodes * (pos_dims + num_classes))

    def forward(self, img, x):
        assert img.ndim == 5
        img = self.pre_layer(img)
        ndown = len(self.downs)
        for i in range(ndown):
            img = self.downs[i](img)
        memory = self.encoder(img)
        output = self.decoder(x, memory)
        output = self.proj(output)
        return output


if __name__ == '__main__':
    from torchinfo import summary
    import json
    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print('Initialize model...')

    img = torch.randn(2, 1, 32, 64, 64)
    seq = torch.randn(2, 10, 8, 4)
    model = NTT(**configs)
    print(model)
    outputs = model(img, seq)

    for output in outputs:
        print('output size: ', output.size())

    summary(model, input_size=[(2, 1, 32, 64, 64), (2, 10, 8, 4)])
