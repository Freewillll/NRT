import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

from models.modules import ConvDropoutNonlinNorm, ConvDropoutNormNonlin

pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

extra_tokens = [pad_token, unk_token, bos_token, eos_token]

PAD = extra_tokens.index(pad_token)
UNK = extra_tokens.index(unk_token)
BOS = extra_tokens.index(bos_token)
EOS = extra_tokens.index(eos_token)


# helpers
def pair3d(t):
    return t if isinstance(t, tuple) else (t, t, t)


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
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        #   q, k, v   dim  b, h, n, d
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #  dots   dim  b, h, n, n
        if attn_mask is not None:
            assert attn_mask.size() == dots.size()
            dots.masked_fill_(attn_mask, -1e9)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# nn.TransformerEncoder
# nn.TransformerDecoder

class TransformerDeconder(nn.Module):
    def __init__(self, input_dim, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.proj = nn.Linear(input_dim, dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, encoder, input_len):
        attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        attn_mask = torch.gt((attn_pad_mask + attn_subsequent_mask), 0)
        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        #    b, n, d
        x = self.proj(x)
        for attn, ff in self.layers:
            x = attn(x, attn_mask) + x
            x = ff(x) + x
        return x


class ViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_depth, image_height, image_width = pair3d(image_size)
        patch_depth, patch_height, patch_width = pair3d(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0,\
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_depth * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1 = patch_depth, p2 = patch_height, p3 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=(3, 3, 3), stride=(1, 1, 1)):
        super(ResidualBlock, self).__init__()
        padding = tuple((k - 1) // 2 for k in kernel)
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ELU(alpha=0.2, inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.elu(out, alpha=0.2, inplace=True)

        return out


class NTT(nn.Module):
    def __init__(self, in_channels, base_num_filters, num_nodes, node_dims, down_kernel_list, stride_list, patch_size,
                 dim, depth, heads, mlp_dim, img_shape):
        super(NTT, self).__init__()
        assert len(down_kernel_list) == len(stride_list)
        self.downs = []

        # the first layer to process the input image
        self.pre_layer = nn.Sequential(
            ConvDropoutNormNonlin(in_channels, base_num_filters, norm_op=nn.BatchNorm3d),
            ConvDropoutNormNonlin(base_num_filters, base_num_filters, norm_op=nn.BatchNorm3d),
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
            self.down_h *= stride[1]
            self.down_w *= stride[2]
            down_filters.append((in_channels, out_channels))
            down = ResidualBlock(in_channels, out_channels, kernel=down_kernel, stride=stride)
            self.downs.append(down)
            in_channels = out_channels
            out_channels = out_channels * 2

        out_channels = int(out_channels / 2)
        *_, d, h, w = img_shape
        d = int(d / self.down_d)
        h = int(h / self.down_h)
        w = int(w / self.down_w)

        self.vit = ViT3D(
            image_size=(d, h, w),
            patch_size=patch_size,
            num_classes=num_nodes * node_dims,
            channels=out_channels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )
        # convert layers to nn containers
        self.downs = nn.ModuleList(self.downs)

    def forward(self, x):
        assert x.ndim == 5
        x = self.pre_layer(x)
        ndown = len(self.downs)
        for i in range(ndown):
            x = self.downs[i](x)
        x = self.vit(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    import json
    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print('Initialize model...')

    input = torch.randn(2, 1, 32, 64, 64)
    model = NTT(**configs)
    print(model)
    outputs = model(input)

    for output in outputs:
        print('output size: ', output.size())

    summary(model, input_size=(2, 1, 32, 64, 64))