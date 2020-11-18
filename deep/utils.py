import os
import shutil
import itertools
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import skimage
import tqdm
from collections import defaultdict


def get_covariance(examples):
    mean, std = examples.mean(dim=0), examples.std(dim=0)
    outer = mean.unsqueeze(0) * mean.unsqueeze(1)
    cov =  (examples.t() @ examples / len(examples) - outer)
    return cov

def calc_weight_alignments(model):
    alignments = {'layers':[]}
    b = model.dfa.feedback_matrix
    bs = []
    for l in range(1,len(model.layers)):
        w = model.state_dict()['layers.{}.weight'.format(l)].view(-1)
        if l == len(model.layers)-1:
            v = b[-1].view(-1)
        else:
            v = (b[l].t()@b[l-1]).view(-1)
        bs.append(v)
        a = v @ w / w.norm() / v.norm()
        alignments['layers'].append(a.item())
    ws = torch.cat([w.view(-1) for w in list(model.state_dict().values())[1:]])
    bs = torch.cat(bs)
    total =  ws @ bs / ws.norm() / bs.norm()
    alignments['total'] = total.item()
    return dict(alignments)

def shuffle_labels(labels, label_noise, num_classes):
    for i in range(len(labels)):
        if np.random.rand()<label_noise:
            labels[i] = np.random.randint(0, num_classes)
    return labels

def get_data(dataset, dataset_path, batch_size, test_batch_size, num_classes=None, input_dim=None, datasize=0, label_noise=0.):
        
    input_sizes = {'MNIST':28, 'FashionMNIST':28, 'CIFAR10':32, 'CIFAR100':32}
    output_sizes = {'MNIST':10, 'FashionMNIST':10, 'CIFAR10':10, 'CIFAR100':100}
    input_channels = {'MNIST':1, 'FashionMNIST':1, 'CIFAR10':3, 'CIFAR100':3}
    input_size = input_sizes[dataset]
    output_size = output_sizes[dataset]
    input_channels = input_channels[dataset]

    dataclass = eval('Fast'+dataset)
    train_data = dataclass(dataset_path, train=True, download=True)
    test_data = dataclass(dataset_path, train=False, download=True)
    
    if num_classes:
        output_size = num_classes
        train_data.targets = train_data.targets%num_classes
        test_data.targets = test_data.targets%num_classes
        
    if datasize is not None:
        train_data.data = train_data.data[:datasize]
        train_data.targets = train_data.targets[:datasize]

    # resizing
    if input_dim is not None:
        input_size = input_dim
        print('Starting the resizing')
        for dataset in [train_data, test_data]:
            size, channels, _, _, = dataset.data.size()
            new_data = torch.empty(size, channels, input_size, input_size)
            for i, img in enumerate(dataset.data):
                for c, img_channel in enumerate(dataset.data[i]): 
                    new_data[i,c] = torch.from_numpy(skimage.transform.resize(img_channel, (input_size, input_size)))
            dataset.data = new_data
        print('Finished the resizing')

    if label_noise:
        for dataset in [train_data, test_data]:
            dataset.targets = shuffle_labels(dataset.targets, label_noise, output_size)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loader_log = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, train_loader_log, test_loader, input_size, output_size, input_channels

def get_random_data(batch_size, test_batch_size, num_classes=None, input_dim=None, datasize=0, label_noise=0., beta=1., alpha=1, task= 'CLASSIFICATION', use_teacher=False):

    assert input_dim**.5 == int(input_dim**.5)
    with torch.no_grad():
        n_batches = max(1,datasize // batch_size)
        n_batches_test = max(1,datasize // test_batch_size)
        train_loader = []
        test_loader = []
        if use_teacher:
            teacher = nn.Sequential(nn.Linear(input_dim, num_classes))
        else:
            if num_classes==2:
                dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.Tensor([[1, alpha*(1-beta)],[alpha*(1-beta), alpha**2]]))
            else:
                raise NotImplementedError
        for i in range(n_batches):
            x = torch.randn(batch_size, input_dim)
            if task=='REGRESSION':
                if use_teacher: y = teacher(x)
                else: y = dist.sample((batch_size,))
                y += label_noise * torch.randn(y.size())
            elif task=='CLASSIFICATION':
                if use_teacher: y = teacher(x).max(1)[1]
                else: raise NotImplementedError
                y = shuffle_labels(y, label_noise, num_classes)
            train_loader.append([x,y])
        for i in range(n_batches_test):
            x = torch.randn(test_batch_size, input_dim)
            if task=='REGRESSION':
                if use_teacher: y = teacher(x)
                else: y = dist.sample((test_batch_size,))
                y += label_noise * torch.randn(y.size())
            elif task=='CLASSIFICATION':
                if use_teacher: y = teacher(x).max(1)[1]
                else: raise NotImplementedError
                y = shuffle_labels(y, label_noise, num_classes)
            test_loader.append([x,y])
    return train_loader, train_loader, test_loader, int(input_dim**.5), num_classes, 1


class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

class FastFashionMNIST(datasets.FashionMNIST):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class FastCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = torch.from_numpy(self.data).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), torch.LongTensor(self.targets).to(self.device)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

class FastCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = torch.from_numpy(self.data).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), torch.LongTensor(self.targets).to(self.device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

#helpers
def copy_py(dst_folder):
    # and copy all .py's into dst_folder
    if not os.path.exists(dst_folder):
        print("Folder doesn't exist!")
        return
    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, dst_folder)
def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))
