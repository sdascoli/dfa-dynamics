# Deep DFA

This folder contains the numerical experiments on deep nonlinear networks.

It makes use of a slightly modified version of the DFA implementation by [1]


## Usage information

To train a fully-connected network, use<br>
```python main.py --model fc --n_layers 5 --hidden_size 100 --activation relu --dataset CIFAR10 --epochs 10```<br>
To train via DFA, use<br>
```python main.py --model fc --n_layers 5 --hidden_size 100 --activation relu --dataset CIFAR10 --epochs 10 --training_method DFA```

To reproduce the data in the paper, use the submit.py files (requires submitit and Slurm)<br>
To produce the plots once the data is generated, use the plots_figures.ipynb notebook

## Requirements

* Torch >= 1.5.

## References

* [1] Launay, Julien, et al. "Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures." Advances in Neural Information Processing Systems 33 (2020).

Code open-sourced at https://github.com/lightonai/dfa-scales-to-modern-deep-learning.