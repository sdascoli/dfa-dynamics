import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mappings import Reshape
from tinydfa import DFA, DFALayer
from tinydfa.alignment import AlignmentMeasurement

class FullyConnected(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=10, training_method='DFA', feedback_init='RANDOM', weight_init='UNIFORM', activation='abs', n_layers=2, seed=1, pseudoinverse=False):
        super(FullyConnected, self).__init__()
        
        self.n_layers = n_layers
        self.training_method = training_method
        self.input_size = input_size
        self.output_size = output_size

        if activation=='relu':
            self.activation = torch.relu
        elif activation=='leaky_relu':
            self.activation = lambda x : torch.relu(x) - 0.01 * torch.relu(-x)
        elif activation=='swish':
            self.activation = lambda x : x / (1+ torch.exp(-x))
        elif activation=='tanh':
            self.activation = torch.tanh
        elif activation=='abs':
            self.activation = torch.abs
        elif activation=='linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError

        self.layers = nn.ModuleList()
        lin = nn.Linear(input_size, hidden_size, bias=False)
        
        self.layers.append(lin)
        for l in range(n_layers-2): 
            lin = nn.Linear(hidden_size, hidden_size, bias=False)
            self.layers.append(lin)

        lin = nn.Linear(hidden_size, output_size, bias=False)
        self.layers.append(lin)

        for layer in self.layers:
            if weight_init == 'GAUSSIAN' :
                nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(layer.weight.size(0)))
            elif weight_init == 'ORTHOGONAL' :
                nn.init.orthogonal_(layer.weight)
            elif weight_init == 'ZERO' :
                nn.init.constant_(layer.weight, 0)

        
        self.training_method = training_method
        if self.training_method in ["DFA", "SHALLOW"]:
            self.dfa_layers = nn.ModuleList()
            for l in range(n_layers-1): 
                self.dfa_layers.append(DFALayer())                

            if feedback_init == 'RANDOM':
                torch.manual_seed(0)
                feedback_matrix = torch.randn(len(self.dfa_layers), output_size, hidden_size)
                torch.manual_seed(seed) #reset the seed same as before
            elif feedback_init == 'UNIFORM':
                torch.manual_seed(0)
                feedback_matrix = 2*torch.rand(len(self.dfa_layers), output_size, hidden_size)-1
                torch.manual_seed(seed) #reset the seed same as before
            elif feedback_init == 'ORTHOGONAL':
                torch.manual_seed(0)
                feedback_matrix = torch.randn(len(self.dfa_layers), output_size, hidden_size)
                nn.init.orthogonal_(feedback_matrix)
                torch.manual_seed(seed) #reset the seed same as before
            elif feedback_init == 'ALIGNED':
                Bs = []
                for i in range(self.n_layers-1):
                    W = copy.deepcopy(self.layers[i+1].weight.data)
                    if pseudoinverse:
                        W = torch.pinverse(W).t()
                    Bl = W
                    for j in range(i+2,self.n_layers):
                        W = self.layers[j].weight.data
                        if pseudoinverse:
                            W = torch.pinverse(W).t()
                        Bl = W @ Bl
                    Bs.append(Bl)
                feedback_matrix = torch.stack(Bs)
            else :
                raise NotImplementedError

            self.dfa = DFA(self.dfa_layers, no_training=(self.training_method == 'SHALLOW'), feedback_matrix=feedback_matrix)

            if weight_init == 'ALIGNED':
                for il, layer in enumerate(self.layers[1:]):
                    if il == len(self.layers)-2:
                        layer.weight.data.copy_(feedback_matrix[-1])
                    else:
                        layer.weight.data.copy_(feedback_matrix[il+1].t() @ feedback_matrix[il])
                    for i, w in enumerate(layer.weight.data):
                        layer.weight.data[i].div_(w.norm())

            

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.training_method in ['DFA', 'SHALLOW']:
            for l in range(self.n_layers-1):
                x = self.layers[l](x)
                x = self.activation(x)
                x = self.dfa_layers[l](x)
            x = self.dfa(self.layers[-1](x))
        else:
            for l in range(self.n_layers-1):
                x = self.layers[l](x)
                x = self.activation(x)
            x = self.layers[-1](x)
        if self.output_size == 1:
            return x.squeeze(1)
        return x


class ConvNet(nn.Module):
    def __init__(self, input_size=[3,28,28], output_size=10, n_layers=2,
                 kernel_size=3, channels=1, 
                 training_method='DFA', feedback_init='RANDOM', weight_init='UNIFORM', activation='abs', stride=1, padding=0, device='cpu'):
        super(ConvNet, self).__init__()
        
        self.n_layers = n_layers
        self.training_method = training_method
        self.input_size = np.array(input_size)
        self.training_method = training_method
        self.channels = channels
        
        if activation=='relu':
            self.activation = lambda : nn.ReLU()
        elif activation=='tanh':
            self.activation = lambda : nn.Tanh()
        elif activation=='linear':
            self.activation = lambda : nn.Identity()
        else:
            raise
        
        self.layers = nn.ModuleList()
        self.dfa_layers = nn.ModuleList()

        conv = nn.Conv2d(self.input_size[0], channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.layers.append(conv)
        self.layers.append(self.activation())
        if self.training_method in ['DFA', 'SHALLOW']:
            dfa_layer = DFALayer()
            self.layers.append(dfa_layer)
            self.dfa_layers.append(dfa_layer)

        new_size = (self.input_size[1:] - kernel_size + 2*padding) // stride + 1
        for l in range(self.n_layers-2):
            conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.layers.append(conv)
            self.layers.append(self.activation())
            if self.training_method in ['DFA', 'SHALLOW']:
                dfa_layer = DFALayer()
                self.layers.append(dfa_layer)
                self.dfa_layers.append(dfa_layer)
            new_size = (new_size - kernel_size + 2*padding) // stride + 1
        
        self.size = int(channels * np.prod(new_size))
        self.layers.append(Reshape([self.size]))
        lin = nn.Linear(self.size, output_size, bias=False)
        self.layers.append(lin)     
        
        if self.training_method in ['DFA', 'SHALLOW']:
            self.dfa = DFA(self.dfa_layers, no_training=(self.training_method == 'SHALLOW'), feedback_matrix = None)
            self.layers.append(self.dfa)
            
        self.sizes = self.get_sizes()
        
    def get_sizes(self):
        # hack to get all sizes
        with torch.no_grad():
            res = []
            han = [l.register_forward_hook(lambda m, i, o: res.append(o.size())) for l in self.layers]
            self.forward(torch.randn(1, *self.input_size))
            [h.remove() for h in han]
        return res 

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
