# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
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
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        # remove dropout, may be not needed
        self.net = nn.Sequential(
            SDPLin(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            SDPLin(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LayerProject(nn.Module):
    def __init__(self, p='fro', radius=1.0, vector=False):
        super().__init__()
        self.radius = radius
        self.p = p
        self.vector = vector
    def forward(self, x):
        if self.vector:
            norm_x = torch.linalg.vector_norm(x, ord=self.p, dim=-1).unsqueeze(1)
        else:
            norm_x = torch.norm(x, p=self.p, dim=(-2,-1)).unsqueeze(1).unsqueeze(1)
        mask = (norm_x < self.radius).to(x.device).float()
        x = mask * x + (1 - mask) * x / norm_x
        
        return x
    def __repr__(self):
        return "LayerProject"

class CenterNorm(nn.Module):

    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=False)
        #self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = normalized_shape/(normalized_shape-1.0)
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = self.scale*(x - u)
        x = self.weight[None, None, :] * x #+ self.bias[None, None, :]
        return x.squeeze()


# def SLL_weight(W, q_param):
#     q_ = q_param[:, None]
#     q = torch.exp(q_)
#     q_inv = torch.exp(-q_)
#     #print(W.shape, q.shape)
#     T = 1/torch.abs(q_inv * W.T @ W * q).sum(1)
#     return W@torch.diag(torch.sqrt(T))

class SLLRes(nn.Module):

  def __init__(self, cin, cout, epsilon=1e-6):
    super(SLLRes, self).__init__()

    self.activation = nn.ReLU(inplace=False)
    self.weights = nn.Parameter(torch.empty(cout, cin))
    self.bias = nn.Parameter(torch.empty(cout))
    self.q = nn.Parameter(torch.rand(cout))

    nn.init.xavier_normal_(self.weights)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)  # bias init

    self.epsilon = epsilon

  def forward(self, x):
    res = nn.functional.linear(x, self.weights, self.bias)
    res = self.activation(res)
    #q_abs = torch.exp(self.q)
    q_ = self.q[None, :]
    q = torch.exp(q_)
    q_inv = torch.exp(-q_)#(1/(q_abs+self.epsilon))[:, None]
    T = 2/torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
    res = T * res
    res = nn.functional.linear(res, self.weights.t())
    out = x - res
    return out

class SDPLin(nn.Module):

  def __init__(self, cin, cout, heads=1, gamma=1.0, epsilon=1e-6, bias=True):
    super(SDPLin, self).__init__()

    self.dim_head = cout//heads
    self.weight = nn.Parameter(torch.empty(heads, self.dim_head, cin))
    nn.init.xavier_normal_(self.weight)
    if bias:
        self.bias = nn.Parameter(torch.empty(cout))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    else: self.bias = None
    self.q = nn.Parameter(torch.rand(heads, cin))

    self.cout = cout
    self.cin = cin
    self.W = None
    self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)
    #self.epsilon = epsilon

#vectorize this operation
  def forward(self, x):
    if self.training or self.W is None:
        
        #self.W = SLL_weight(self.weight, self.q).reshape(self.cout, self.cin)
        q_ = self.q[:,:,None]
        q = torch.exp(q_)
        q_inv = torch.exp(-q_)
        T = 1/torch.abs(q_inv*torch.transpose(self.weight,1,2)@self.weight*q/self.gamma**2).sum(2)
        self.W = (self.weight@torch.diag_embed(torch.sqrt(T))).view(self.cout, self.cin)

    W = self.W if self.training else self.W.detach()

    out =  nn.functional.linear(x, W, self.bias)
    return out



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

class SDPConv(nn.Module):

  def __init__(self, input_size, cin, cout, kernel_size=3, epsilon=1e-6):
    super(SDPConv, self).__init__()

    self.activation = nn.ReLU(inplace=False)

    self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
    self.bias = nn.Parameter(torch.empty(cout))
    self.q = nn.Parameter(torch.randn(cout))

    nn.init.xavier_normal_(self.kernel)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound) # bias init

    self.epsilon = epsilon

  def forward(self, x):
    res = F.conv2d(x, self.kernel, bias=self.bias, padding=1)
    res = self.activation(res)
    batch_size, cout, x_size, x_size = res.shape
    kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
    q_abs = torch.abs(self.q)
    T = 2 / (torch.abs(q_abs[None, :, None, None] * kkt).sum((1, 2, 3)) / q_abs)
    res = T[None, :, None, None] * res
    res = F.conv_transpose2d(res, self.kernel, padding=1)
    out = x - res
    return out  

class OrthogonLin(nn.Linear):
    def __init__(self, in_features, out_features, heads=8, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        print(self.weight.shape, in_features, out_features)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None
        self.heads = heads
        self.dim_head = in_features//heads
        
    def forward(self, x):
        if self.training or self.Q is None:
            Q_list = []
            #This is probably very inefficient
            for j in range(self.heads):
                Wj = self.weight[j*self.dim_head:j*self.dim_head+self.dim_head,:]
                #print('Wj shape', Wj.shape)
                Qj = cayley(self.alpha * Wj / Wj.norm())
                Q_list.append(Qj)
            self.Q = torch.vstack(Q_list)
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
        self.to_q = SDPLin(dim, inner_dim, heads=heads, bias=False)
        self.to_k = SDPLin(dim, inner_dim, heads=heads, bias=False)
        self.to_v = SDPLin(dim, inner_dim, heads=heads, bias=False)
        # self.to_q = nn.Linear(dim, inner_dim , bias = False)
        # self.to_k = nn.Linear(dim, inner_dim , bias = False)
        #self.to_v = nn.Identity()

        #print('dims of Wo', inner_dim, dim)
        self.to_out = SDPLin(inner_dim, dim, heads=1)
        #) if project_out else nn.Identity()

    def forward(self, x):
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
        #print('pre Wo out shape', out.shape)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        id_map = nn.Identity()
    
        # softmax_project = nn.Sequential(nn.Flatten(start_dim=1),
        #                                 nn.Softmax(dim=-1),
        #                                 nn.Unflatten(1,(num_patches+1, dim)))
        for j in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LayerProject(), Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, CenterNorm(dim), SLLRes(dim, mlp_dim))
            ]))


    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)
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
            SDPLin(patch_dim, dim),
        )
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) c p1 p2', p1 = patch_height, p2 = patch_width),
        #     SDPConv(patch_dim, dim),
        #     SDPConv()
        # )  

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #try without dropout, may be not needed
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_patches, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            CenterNorm(dim),
            #nn.LayerNorm(dim),
            SDPLin(dim, num_classes)
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
