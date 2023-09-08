import torch
from models.vit import ViT
import argparse
from data_utils import get_dataset
from collections import OrderedDict
import numpy as np

from torch import nn


class ViTAnalyzer(nn.Module):
    def __init__(self, net, batch_size=100, eps=0.1):
        super(ViTAnalyzer, self).__init__()
        # extract components
        self.net = net
        self.layers = 6
        self.heads = 8
        self.n_patchs = 65
        self.eps = eps
        self.batch_size = batch_size
        self.selected_outputs = {}
        for l in range(self.layers):
            self.net.transformer.layers[l][0].norm.register_forward_hook(self.getActivation('x_{}'.format(l)))
            self.net.transformer.layers[l][0].fn.attend.register_forward_hook(self.getActivation('Px_{}'.format(l)))
    
    def getActivation(self, name):
    # the hook signature
        def hook(model, input, output):
            self.selected_outputs[name] = output.detach()
        return hook

    def compute_bound_layer(self, Px, x, eps):
        d = x.shape[-1]
        Px_norm = torch.linalg.matrix_norm(Px, ord=2, dim=(-2,-1))
        x_norm = torch.norm(x, p='fro', dim=(-2,-1)).unsqueeze(-1)
        beta = eps*torch.sqrt(2/d*(4*torch.square(x_norm) + eps**2)) * (x_norm + eps)
        delta_heads = eps*Px_norm + beta
        return torch.sum(delta_heads, -1, keepdim=True)

    def compute_bound(self):
        delta = torch.ones(self.batch_size,1)*self.eps
        for l in range(self.layers):
            Px = self.selected_outputs['Px_{}'.format(l)]
            x = self.selected_outputs['x_{}'.format(l)]
            delta = delta + self.compute_bound_layer(Px, x, delta)
        return delta

    def forward(self, x, y):
        y_hat = self.net(x)
        delta = self.compute_bound()
        return y_hat, delta




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()
args.dataset = 'c10'
args.rcpaste = True
args.autoaugment = True
imsize = int(args.size)
size = imsize
batch_size = 10
eps = 0.1
# ViT for cifar10
net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.0,
    emb_dropout = 0.0
)

checkpoint = torch.load('./checkpoint/{}-{}-ckpt.t7'.format(args.net, args.patch), map_location=torch.device('cpu'))
state_dict = checkpoint['model']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

net.load_state_dict(new_state_dict)
net.eval()
trainset, testset = get_dataset(args)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

X, Y = next(iter(testloader))

va = ViTAnalyzer(net, batch_size=batch_size, eps=eps)
Y, deltas = va(X, Y)
print(deltas)







