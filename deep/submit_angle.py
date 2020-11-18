import torch
import numpy as np
import submitit
from pathlib import Path
from main import main
import collections
import itertools
import time
import os
from utils import copy_py, dict_product
import shutil

#get default arguments
from config import add_arguments
import argparse
from argparse import Namespace
def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults
parser = argparse.ArgumentParser()
parser = add_arguments(parser)
args = get_argparse_defaults(parser)

#create folder
folder = 'r.{}'.format(int(time.time()))
if not os.path.exists(folder):
    os.mkdir(folder)
copy_py(folder) 

grid = collections.OrderedDict({
    'seed' : range(10)],
    'hidden_size' : [64],
    'n_layers': [3],
    'activation' : ['relu', 'tanh'],
    'dataset' : ['MNIST', 'CIFAR10'],
    'epochs' : [1000],
    'n_saves' : [40],
    'training_method' : ['BP', 'DFA'],
    'feedback_init' : ['UNIFORM'],
    'weight_init': ['UNIFORM'],
    'learning_rate' : [0.001],
    'momentum' : [0.],
    'batch_size' : [32],
    'test_batch_size' : [1000],
    'model' : ['fc'],
    'no_gpu' : [False],
    'dataset_path' : ['~/data'],
    'datasize' : [None],
    'num_classes' : [10],
    'label_noise' : [0., .1, .2, .5, .9],
    'input_dim' : [14],
    'optimizer' : ['SGD'],
    'task' : ['CLASSIFICATION']
    })


torch.save(grid, folder + '/params.pkl')

ex = submitit.AutoExecutor(folder)
ex.update_parameters(mem_gb=10, nodes=1, cpus_per_task=8, gpus_per_node=1, timeout_min=1440)
jobs = []
with ex.batch():
    for i, params in enumerate(dict_product(grid)):
        params['name'] = folder+'/{:06d}'.format(i)
        for k,v in params.items():
            args[k] = v
        job = ex.submit(main, Namespace(**args))
        jobs.append(job)

