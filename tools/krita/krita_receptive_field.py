# Receptive field of the two shipped graphs, rebuilt from the layer specs read off the .xml (no foreign code imported).
import torch, torch.nn as nn
C=lambda i,o,k,s,p: nn.Conv2d(i,o,k,s,p)
T=lambda i,o: nn.ConvTranspose2d(i,o,4,2,1)
sk=[C(1,48,5,2,2),C(48,128,3,1,1),C(128,256,3,1,1),T(256,256),C(256,128,3,1,1),T(128,128),C(128,48,3,1,1),T(48,48),C(48,24,3,2,1),C(24,1,3,2,1)]
sm=[C(1,48,5,1,2),C(48,128,9,2,4),C(128,256,5,1,2),C(256,256,5,1,2),C(256,256,9,2,4),C(256,256,3,1,1),C(256,512,3,1,1),C(512,256,3,1,1),C(256,256,3,1,1),C(256,128,3,1,1),T(128,128),C(128,128,3,1,1),C(128,48,3,1,1),T(48,48),C(48,24,3,1,1),C(24,1,3,1,1)]
for name,layers in [('Sketchy',sk),('Smooth',sm)]:
    net=nn.Sequential(*layers)
    for p in net.parameters(): nn.init.constant_(p,0.01)
    n=256; x=torch.zeros(1,1,n,n,requires_grad=True)
    y=net(x); assert y.shape==x.shape, y.shape
    y[0,0,n//2,n//2].backward()
    g=(x.grad[0,0].abs()>0).nonzero()
    print(name,'params',sum(p.numel() for p in net.parameters()),'RF rows',g[:,0].min().item(),g[:,0].max().item(),'width',g[:,0].max().item()-g[:,0].min().item()+1)
