import numpy as np
import copy
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tinydfa import DFA, DFALayer
from tinydfa.alignment import AlignmentMeasurement
from models import *
from mappings import cnn2fc
from utils import get_data, get_random_data, calc_weight_alignments, get_covariance
from config import add_arguments
import pdb

def train_and_test(train_loader, train_loader_log, test_loader, model, optimizer, crit, device, epoch, alignment, verbose=True, log_every=100, args=None):
    model.train()
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    weight_alignments = []
    grad_alignments = []

    for b, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()
        if args.grad_perturbation:
            for l, layer in enumerate(model.layers):
                if layer.__class__ in [nn.Linear, nn.Conv2d]:
                    grad_norm = layer.weight.grad.norm()
                    if args.perturbation_type=='FIXED':
                        perturbation = torch.ones_like(layer.weight.data)
                    elif args.perturbation_type=='RANDOM':
                        perturbation = torch.randn_like(layer.weight.data)
                    else:
                        raise
                    perturbation = args.grad_perturbation*perturbation/perturbation.norm()*grad_norm
                    new_grad = layer.weight.grad+perturbation
                    # print(torch.dot(layer.weight.grad.view(-1), new_grad.view(-1))/layer.weight.grad.norm()/new_grad.norm())
                    model.layers[l].weight.data = layer.weight.data - args.learning_rate * perturbation
                else: pass

        if b % log_every == 0:
            test_loss, test_acc = test(test_loader, model, device, crit, epoch)
            train_loss, train_acc = test(train_loader_log, model, device, crit, epoch)
            if verbose : print("Epoch {}, Batch {} : tr loss {:.6f}, test loss {:.6f}, train acc {:.6f}, test acc {:.6f}".format(epoch,b,train_loss,test_loss,train_acc,test_acc))
            if model.training_method == "DFA":
                grad_alignment, difference = alignment.measure_alignment(data, target, crit)
                if verbose :
                    print("Grad   alignment : {0:.6f}. Layerwise : {1}".format(grad_alignment['total'], grad_alignment['layers']))
                grad_alignments.append(grad_alignment)
                if args.model=='fc':
                    weight_alignment = calc_weight_alignments(model)
                    if verbose:
                        print("Weight   alignment : {0:.6f}. Layerwise : {1}".format(weight_alignment['total'], weight_alignment['layers']))
                    weight_alignments.append(weight_alignment)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

    return train_losses, test_losses, train_accs, test_accs, weight_alignments, grad_alignments

def test(test_loader, model, device, crit, epoch):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for b, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += crit(output, target).item()
            if isinstance(crit, nn.MSELoss):
                pred, correct= 0, 0
            else:
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
    loss /= total 
    acc = correct / total * 100
    model.train()
    return loss, acc
        
def main(args):
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    torch.manual_seed(args.seed)

    if args.activation == 'linear':
        args.weight_init = 'ZERO'
    else:
        args.weight_init = 'UNIFORM'

    if args.dataset != 'RANDOM':
        train_loader, train_loader_log, test_loader, input_size, output_size, input_channels = get_data(args.dataset, args.dataset_path, args.batch_size, args.test_batch_size, num_classes=args.num_classes, input_dim=args.input_dim, datasize=args.datasize, label_noise=args.label_noise)
    else:
        train_loader, train_loader_log, test_loader, input_size, output_size, input_channels = get_random_data(batch_size=args.batch_size, test_batch_size=args.test_batch_size, num_classes=args.num_classes, input_dim=args.input_dim, datasize=args.datasize, label_noise=args.label_noise, beta=args.beta, alpha=args.alpha, task=args.task)        
        
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    feedback_matrix = []
    weights = []
    weights = {}
    weight_alignments = {'total':[]}
    grad_alignments = {'total':[]}
    for l in range(args.n_layers-1):
        weight_alignments[l]=[]
        grad_alignments[l]=[]

    if args.model == 'fc':
        model = FullyConnected(input_size = input_channels * input_size**2,
                               output_size=output_size, hidden_size=args.hidden_size, n_layers=args.n_layers,
                               training_method=args.training_method, activation=args.activation, 
                               feedback_init=args.feedback_init, weight_init = args.weight_init,
                               seed=args.seed).to(device)
    elif args.model == 'cnn':
        model = ConvNet(input_size = [input_channels, input_size, input_size], output_size = output_size,
                        kernel_size = args.kernel_size, channels=args.hidden_size, n_layers=args.n_layers,
                        training_method=args.training_method, activation=args.activation, 
                        feedback_init=args.feedback_init, weight_init = args.weight_init, device=device).to(device)

        if args.training_method == 'DFA':
            if args.feedback_init == 'ALIGNED_NMF':
                from sklearn.decomposition import NMF
                efcn = cnn2fc(model, return_mask=True)
                print("Converted to an eFCN")
                layers = list(efcn.state_dict().values())
                for l, w in enumerate(layers):
                    w = 1. - w # take the inverse of the mask
                    if l==len(layers)-1 or l==0: continue
                    nmf = NMF(n_components=output_size, init='random', random_state=0)
                    W = torch.from_numpy(nmf.fit_transform(w)).t()
                    H = torch.from_numpy(nmf.components_)
                    model.dfa.feedback_matrix[l][:,:W.size(-1)] = W
                    if l==1:
                        model.dfa.feedback_matrix[l-1][:,:H.size(-1)] = H
                    else:
                        model.dfa.feedback_matrix[l-1][:,:H.size(-1)] += H
                        model.dfa.feedback_matrix[l-1][:,:H.size(-1)] /= 2                        
                model.layers[-2].weight.data.copy_(W)
                model.layers[-2].weight.requires_grad=True
                print("Finished the matrix factorization")
            if args.feedback_init == 'ALIGNED_SVD':
                efcn = cnn2fc(model, return_mask=False)
                print("Converted to an eFCN")
                layers = list(efcn.state_dict().values())
                for l, w in enumerate(layers):
                    if l==len(layers)-1 or l==0 : continue
                    U,S,V = torch.svd(w)
                    U, S, V = U[:, :output_size], S[:output_size], V[:, :output_size]
                    U = U @ torch.diag(S**.5)
                    V = V @ torch.diag(S**.5)
                    model.dfa.feedback_matrix[l][:,:U.size(0)] = U.t()
                    if l==1:
                        model.dfa.feedback_matrix[l-1][:,:V.size(0)] = V.t()
                    else:
                        model.dfa.feedback_matrix[l-1][:,:V.size(0)] += V.t()
                        model.dfa.feedback_matrix[l-1][:,:V.size(0)] /= 2
                model.layers[-2].weight.data.copy_(U.t())
                print("Finished the matrix factorization")
            if args.feedback_init == 'REPEATED':
                for l in range(len(model.dfa.feedback_matrix)-1):
                    repeat = model.dfa.feedback_matrix[l].size(1)//args.hidden_size
                    for o in range(output_size):
                        for c in range(args.hidden_size):
                            model.dfa.feedback_matrix[l][o,c*repeat:(c+1)*repeat] = model.dfa.feedback_matrix[l][o,c*repeat].repeat(repeat)

            for l in range(len(model.dfa.feedback_matrix)):
                for col in range(len(model.dfa.feedback_matrix[l])):
                    model.dfa.feedback_matrix[l][col]/=model.dfa.feedback_matrix[l][col].norm()        
        
    else:
        raise NotImplementedError

    print(args)
    print(model)
    alignment = AlignmentMeasurement(model, device) if args.training_method != "BP" else None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError

    if args.task=='CLASSIFICATION':
        crit = nn.CrossEntropyLoss(reduction='sum')
    else:
        crit = nn.MSELoss(reduction='sum')

    checkpoints = np.unique(np.logspace(0, np.log10(args.epochs), args.n_saves).astype(int))
        
    weights[0] = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        train_loss, test_loss, train_acc, test_acc, weight_alignment, grad_alignment = train_and_test(train_loader, train_loader_log, test_loader, model, optimizer, crit, device, epoch, alignment, verbose=True, log_every=args.log_every, args=args)
        for l in range(args.n_layers-1):
            weight_alignments[l]+=[x['layers'][l] for x in weight_alignment]
            grad_alignments[l]+=[x['layers'][l] for x in grad_alignment]
        weight_alignments['total']+=[x['total'] for x in weight_alignment]
        grad_alignments['total']+=[x['total'] for x in grad_alignment]
        train_losses+=train_loss
        test_losses+=test_loss
        train_accs+=train_acc
        test_accs+=test_acc
        if epoch in checkpoints:
            weights[epoch] = copy.deepcopy(model.state_dict())
    
        if args.training_method != 'BP': feedback_matrix=model.dfa.feedback_matrix
        run = {'args':args, 'test_loss': test_losses, 'train_loss': train_losses, 'train_accs':train_accs, 'test_accs': test_accs, 'weight_alignments':weight_alignments, 'grad_alignments':grad_alignments,  'weights':weights, 'feedback_matrix':feedback_matrix, 'finished': False}
        torch.save(run, args.name + '.pyT') # overwrite

    weights[epoch] = copy.deepcopy(model.state_dict())
    run = {'args':args, 'test_loss': test_losses, 'train_loss': train_losses, 'train_accs':train_accs, 'test_accs': test_accs, 'weight_alignments':weight_alignments, 'grad_alignments':grad_alignments, 'weights':weights, 'feedback_matrix':feedback_matrix, 'finished': True}
    torch.save(run, args.name + '.pyT')

        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DFA experiments')
    parser = add_arguments(parser)
    args = parser.parse_args()    

    main(args)
