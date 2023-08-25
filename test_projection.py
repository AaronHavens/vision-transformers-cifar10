import torch
from models.vit import Projection

X = torch.rand((10,3))
F = Projection(p=2, radius=1.0)

Z = F(X)

print(torch.linalg.vector_norm(X,ord=2,dim=1))
print(torch.linalg.vector_norm(Z,ord=2,dim=1))