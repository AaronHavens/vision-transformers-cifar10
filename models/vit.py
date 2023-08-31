# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, norm_fn, fn):
        super().__init__()
        self.norm = norm_fn#nn.LayerNorm(dim)
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


class LayerProject(nn.Module):
    def __init__(self, p=2, radius=1.0):
        super().__init__()
        self.radius = radius
        self.p = p

    def forward(self, x):
        norm_x = torch.linalg.vector_norm(x, ord=self.p, dim=1).unsqueeze(1)
        mask = (norm_x < self.radius).to(x.device).float()
        x = mask * x + (1 - mask) * x / norm_x
        return x
    def __repr__(self):
        return "LayerProject"

class CenterNorm(nn.Module):

    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = normalized_shape/(normalized_shape-1.0)
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = self.scale*(x - u)
        x = self.weight[None, None, :] * x + self.bias[None, None, :]
        return x.squeeze()


# from www.github.com/acfr/LBDN
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

def cayley_square(W):
    S = W - W.T
    I = torch.eye(W.shape[0]).to(W.device)
    return 2*torch.inverse(I+S) - I

class OrthogonLin(nn.Linear):
    def __init__(self, in_features, out_features, heads=8, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None
        self.heads = heads
        self.dim_head = in_features//heads
        
    def forward(self, x):
        if self.training or self.Q is None:
            Q_list = []
            print('W shape', self.weight.shape, self.heads)
            for j in range(self.heads):
                Wj = self.weight[j*self.dim_head:j*self.dim_head+self.dim_head,:]
                print('Wj shape', Wj.shape)
                Qj = cayley(self.alpha * Wj / Wj.norm())
                Q_list.append(Qj)
            self.Q = torch.vstack(Q_list) # need to put on device i think?
        Q = self.Q if self.training else self.Q.detach()
        y = nn.functional.linear(self.scale * x, Q, self.bias)
        return y

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = OrthogonLin(dim, inner_dim, heads=heads, bias=False)
        self.to_k = OrthogonLin(dim, inner_dim, heads=heads, bias=False)
        self.to_v = OrthogonLin(dim, inner_dim, heads=heads, bias=False)
        # self.to_q = nn.Linear(dim, inner_dim , bias = False)
        # self.to_k = nn.Linear(dim, inner_dim , bias = False)
        # self.to_v = nn.Linear(dim, inner_dim , bias = False)

        print('dims of Wo', inner_dim, dim)
        self.to_out = nn.Sequential(
            OrthogonLin(inner_dim, dim, heads=heads),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #print(x.shape)
        #qkv = self.to_qkv(x).chunk(3, dim = -1)
        q_ = self.to_q(x)
        k_ = self.to_k(x)
        v_ = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q_,k_,v_))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        #print(out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')
        print('pre Wo out shape', out.shape)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CenterNorm(dim), Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, CenterNorm(dim), FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
            # self.layers.append(nn.ModuleList([
            #     PreProject(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            #     PreProject(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            # ]))


    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            CenterNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
