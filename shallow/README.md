# fa++: Training two-layer neural networks with direct feedback alignment

This package provides utilities to simulate learning in fully connected
two-layer neural networks trained under both backpropagation (BP) and direct feedback aligment (DFA).
It was used to perform the shallow network experiment experiments of our
recent paper on the performances of direct feedback alignment in a variety of settings.
The package contains three .cpp codes:

(1) fa.cpp ## used to run the training of a 2 layer neural network (2LNN) with both BP and DFA
(2) fa_ode.cpp ## integrate the ordinary diferential equations (ODE) describing DFA dynamics.
(3) bp_ode.cpp [1] ## integrate the ordinary diferential equations (ODE) describing BP dynamics.

## Usage information

To build the simulator, simply type
```
make fa.exe
```
which will create a file ``fa.exe``. 

Run ``./fa.exe -h`` for usage information and optinal arguments.

Example of command to train a 2LNN of K=10 hidden nodes on the output of a techer of M=2 hidden nodes with BP
``./fa.exe --K 10 --M 2 --lr 0.1``

To train with DFA simply add the ``--fa`` option.

To save initial weight configuration to a file add the option ``--save``.

The output of the program is a ``.dat`` file containing the generalisation error of the network at increasing training times.

Likewise, to build the ODE integrators type
```
make fa_ode.exe
make bp_ode.exe 
```

Run ``./fa_ode.exe -h`` for usage information.

To check agreement between ODEs and simulations, run a simulaton with the ``--save`` option, this will save the initial weights of the network to a file ``file.dat``.

Then run either ``./fa_ode.exe`` or ``./bp_ode.exe`` with the command ``--prefix file`` which integrates the ODEs starting from initial condition given by the weights in ``file.dat``. 

## Requirements

* All linear algebra operations are implemented using
  [Armadillo](http://arma.sourceforge.net/), a C++ library for linear algebra &
  scientific computing

## References

* [1] This code was implemented by S. Goldt in the paper:
S. Goldt, M.S. Advani, A.M. Saxe, F. Krzakala, L. Zdeborová, NeurIPS 2019,
[arXiv:1906.08632](https://arxiv.org/abs/1906.08632)
  For full reference please see https://github.com/sgoldt/nn2pp
