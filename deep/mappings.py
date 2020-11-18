# Mapping functions / input nets must follow certain conventions
import copy
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import math

class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape # must be a list

    def __repr__(self):
        return ('Reshape({})'.format(self.shape))    
        
    def forward(self, x):
        self.bs = x.size(0)
        return x.view(self.bs, *self.shape)

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.net = nn.Sequential(*layers)


def cnn2fc(CNN, return_mask=False):
    # Given CNN returns a sequentially defined FC network with pooling/padding layers...
    # if return_mask is on, return a FC model where the zero blocks have ones 
    net_sq = list(CNN.named_children())[0][1]

    sizes = [CNN.input_size] + CNN.sizes
    size_pairs = [(sizes[i - 1][-1], sizes[i][-1]) for i in range(1, len(sizes))]

    layers = []
    masks = []
    for size, module in zip(size_pairs, net_sq):
        
        if module.__class__ == nn.Conv2d:
            assert module.padding == (0, 0)
            d_in, d_out = size
            k, s = module.kernel_size[0], module.stride[0]
            ch_in, ch_out = module.in_channels, module.out_channels
            lin_in, lin_out = ch_in*d_in*d_in, ch_out*d_out*d_out

            if return_mask:
                lin_W = torch.ones(lin_out, lin_in)
            else:
                lin_W = torch.zeros(lin_out, lin_in)

            for idx in range(ch_out):
                for i in range(d_out):
                    for j in range(d_out):
                        reverse_map = lin_W.view(ch_out, d_out, d_out, ch_in, d_in, d_in)[idx][i, j, :, :]
                        if return_mask:
                            reverse_map[:, i*s:i*s+k, j*s:j*s+k]=torch.zeros_like(module.weight[idx])
                        else:
                            reverse_map[:, i*s:i*s+k, j*s:j*s+k].copy_(module.weight[idx])
            
            with_bias = module.bias is not None
            if with_bias:
                if return_mask:
                    lin_B = torch.zeros(d_out * d_out * ch_out)
                else:
                    lin_B = module.bias.expand(d_out * d_out, ch_out).t().contiguous().view(lin_out)
                
            lin = nn.Linear(lin_in, lin_out, bias=with_bias)
            lin.weight.data.copy_(lin_W)

            if with_bias:
                lin.bias.data.copy_(lin_B)

            # append layers
            layers.append(Reshape([lin_in]))
            layers.append(lin)
            layers.append(Reshape([ch_out, d_out, d_out]))

        elif module.__class__ == nn.Linear:
            if return_mask:
                new_module = copy.deepcopy(module)
                new_module.weight.data.zero_()
                if new_module.bias is not None:
                    new_module.bias.data.zero_()
                layers.append(new_module)
            else:
                layers.append(copy.deepcopy(module)) 
                
        else:
            layers.append(copy.deepcopy(module)) 
            
    return nn.Sequential(*layers)
    
def cnn2lc(CNN):
    # Given CNN returns a sequentially defined LC network with pooling/padding layers...
    net_sq = list(CNN.named_children())[0][1]

    sizes = [CNN.input_size] + CNN.sizes
    size_pairs = [(sizes[i - 1][-1], sizes[i][-1]) for i in range(1, len(sizes))]

    layers = []
    for size, module in zip(size_pairs, net_sq):
        
        if module.__class__ == nn.Conv2d:

            with_bias = module.bias is not None

            assert module.padding == (0, 0)
            d_in, d_out = size
            k, s = module.kernel_size[0], module.stride[0]
            ch_in, ch_out = module.in_channels, module.out_channels

            lc_W = torch.zeros(1, ch_out, ch_in, d_out, d_out, k**2)
            
            for idx in range(ch_out):
                for i in range(d_out):
                    for j in range(d_out):
                        lc_W[0, idx, :, i, j, :].copy_(module.weight[idx].view(ch_in, k**2))
                                    
            if with_bias:
                lc_B = module.bias.expand(d_out * d_out, ch_out).t().contiguous().view(1, ch_out, d_out, d_out)
                
            # lc = Conv2dLocal(in_channels=ch_in, out_channels=ch_out, in_height=d_in, in_width=d_in, kernel_size=k, stride=s, bias=with_bias)
            lc = LocallyConnected2d(ch_in, ch_out, d_out, k, s, with_bias)
            lc.weight.data.copy_(lc_W.view_as(lc.weight.data))

            if with_bias:
                lc.bias.data.copy_(lc_B.view_as(lc.bias.data))

            # append layers
            # layers.append(Reshape([lin_in]))
            layers.append(lc)
            # layers.append(Reshape([ch_out, d_out, d_out]))

        elif module.__class__ in [nn.ReLU,nn.Dropout,Reshape,nn.MaxPool2d,nn.ZeroPad2d,nn.Linear]:
            layers.append(copy.deepcopy(module)) 

    return nn.Sequential(*layers)

def lc2fc(LCN, return_mask=False):
    # Given a LCN returns a sequentially defined FC network with pooling/padding layers...
    # if return_mask is on, return a FC model where the zero blocks have ones 
    net_sq = [m[1] for m in LCN.net.named_children()]
    
    layers = []
    masks = []
    for module in net_sq:
        
        if module.__class__ == LocallyConnected2d:
            k, s = module.kernel_size[0], module.stride[0]
            ch_in, ch_out = module.in_channels, module.out_channels
            d_out = module.output_size[0]
            d_in = s * (d_out-1) + k
            lin_in, lin_out = ch_in*d_in*d_in, ch_out*d_out*d_out

            if return_mask:
                lin_W = torch.ones(lin_out, lin_in)
            else:
                lin_W = torch.zeros(lin_out, lin_in)

            weight = module.weight.squeeze(0).view(ch_out, ch_in, d_out, d_out, k, k)
            for idx in range(ch_out):
                for i in range(d_out):
                    for j in range(d_out):
                        reverse_map = lin_W.view(ch_out, d_out, d_out, ch_in, d_in, d_in)[idx][i,j, :, :, :]
                        if return_mask:
                            reverse_map[:, i*s:i*s+k, j*s:j*s+k]=torch.zeros_like(weight[idx,:,i,j, :, :])
                        else:
                            reverse_map[:, i*s:i*s+k, j*s:j*s+k].copy_(weight[idx,:,i,j, :, :])
            
            with_bias = module.bias is not None
            if with_bias:
                if return_mask:
                    lin_B = torch.zeros(d_out * d_out * ch_out)
                else:
                    lin_B = module.bias.view(lin_out)
                
            lin = nn.Linear(lin_in, lin_out, bias=with_bias)
            lin.weight.data.copy_(lin_W)

            if with_bias:
                lin.bias.data.copy_(lin_B)

            # append layers
            layers.append(Reshape([lin_in]))
            layers.append(lin)
            layers.append(Reshape([ch_out, d_out, d_out]))

        elif module.__class__ in [nn.ReLU,nn.Dropout,Reshape,nn.MaxPool2d,nn.ZeroPad2d]:
            layers.append(copy.deepcopy(module)) 

        elif module.__class__ == nn.Linear:
            if return_mask:
                new_module = copy.deepcopy(module)
                new_module.weight.data.zero_()
                new_module.bias.data.zero_()
                layers.append(new_module)
            else:
                layers.append(copy.deepcopy(module)) 
    return nn.Sequential(*layers)

class Decomposed_LC(nn.Module):
    def __init__(self, net, factor=2):
        super().__init__()
        net_sq = [m[1] for m in list(net.named_children())]
        layers = []
        self.factor = factor
        for module in net_sq:
            if module.__class__ == LocallyConnected2d:
                new_ch = int(module.out_channels/factor)
                with_bias = module.bias is not None
                new_layer = nn.ModuleList([LocallyConnected2d(module.in_channels, 
                                                              new_ch,
                                                              module.output_size[0], 
                                                              module.kernel_size[0],
                                                              module.stride[0],
                                                              bias = with_bias)
                                           for i in range(factor)])
                for i, sublayer in enumerate(new_layer):
                    if with_bias:
                        sublayer.bias.data.copy_  (module.bias.data[:,i*new_ch:(i+1)*new_ch])
                    sublayer.weight.data.copy_(module.weight.data[:,i*new_ch:(i+1)*new_ch])
                layers.append(new_layer)
            else:
                layers.append(module)
        self.net = nn.Sequential(*layers)
                
    def forward(self, x):
        for i, layer in enumerate(self.net):
            if layer.__class__ == nn.modules.container.ModuleList:
                x = torch.cat( [sublayer(x) for sublayer in layer], dim=1)
            else:
                x = layer(x)
        return x

class Recomposed_LC(nn.Module):
    def __init__(self, net):
        super().__init__()
        factor = net.factor
        net_sq = [m[1] for m in list(net.net.named_children())]
        layers = []
        for module in net_sq:
            if module.__class__ == nn.ModuleList:
                first = module[0]
                old_ch = first.out_channels
                with_bias = first.bias is not None
                layer = LocallyConnected2d(first.in_channels,
                                           old_ch*factor,
                                           first.output_size[0],
                                           first.kernel_size[0],
                                           first.stride[0],
                                           bias=with_bias)
                for i, submodule in enumerate(module):
                    if with_bias:
                        layer.bias.data[:,i*old_ch:(i+1)*old_ch].copy_(submodule.bias.data)
                    layer.weight.data[:,i*old_ch:(i+1)*old_ch].copy_(submodule.weight.data)
                layers.append(layer)
            else:
                layers.append(copy.deepcopy(module))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


