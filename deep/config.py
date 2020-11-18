def add_arguments(parser):
    parser.add_argument('-t', '--training_method', type=str, choices=['BP', 'DFA', 'SHALLOW'], default='BP',
                        metavar='T', help='training method to use, choose from backpropagation (BP), direct feedback '
                                          'alignment (DFA), or only topmost layer (SHALLOW) (default: DFA)')
    parser.add_argument('-fi', '--feedback_init', type=str, choices=['RANDOM', 'UNIFORM', 'ALIGNED_NMF', 'ALIGNED_SVD', 'ORTHOGONAL', 'REPEATED'], default='RANDOM',
                        metavar='T', help='init method for the feedback matrices')
    parser.add_argument('-wi', '--weight_init', type=str, choices=['GAUSSIAN', 'UNIFORM', 'ORTHOGONAL', 'ALIGNED', 'ZERO'], default='UNIFORM',
                        metavar='T', help='init method for the weights')
    
    parser.add_argument('-d', '--dataset', type=str, choices=['MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'RANDOM'], default='MNIST', metavar='D',
                        help='dataset choice')
    parser.add_argument('--datasize', type=int, default=None, metavar='D',
                        help='training dataset size')
    parser.add_argument('--input_dim', type=int, default=None, metavar='D',
                        help='resize image')
    parser.add_argument('--num_classes', type=int, default=None, metavar='D',
                        help='number of classes')
    parser.add_argument('--alpha', type=float, default=1., metavar='D',
                        help='variance of second class in random regression experiment')
    parser.add_argument('--beta', type=float, default=1., metavar='D',
                        help='correlation parameter between two classes in random regression experiment')
    parser.add_argument('--label_noise', type=float, default=0, metavar='D',
                        help='number between 0 and 1')
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='B',
                        help='training batch size (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='B',
                        help='testing batch size (default: 1000)')

    parser.add_argument('--model', type=str, choices=['fc','cnn'], default='fc', metavar='H',
                        help='model architecture')
    parser.add_argument('--task', type=str, choices=['CLASSIFICATION','REGRESSION'], default='CLASSIFICATION', metavar='H',
                        help='classification or regression')
    parser.add_argument('--hidden_size', type=int, default=64, metavar='H',
                        help='hidden layer size (default: 256)')
    parser.add_argument('--kernel_size', type=int, default=4, metavar='H',
                        help='filter size for CNN (default: 4)')
    parser.add_argument('--n_layers', type=int, default=3, metavar='H',
                        help='number of hidden layers')
    parser.add_argument('--activation', type=str, choices=['linear', 'relu', 'tanh', 'leaky_relu', 'swish'], default='relu', metavar='H',
                        help='activation function')

    parser.add_argument('-e', '--epochs', type=int, default=15, metavar='E',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--log_every', type=int, default=100, metavar='E',
                        help='frequency of logs in steps')
    parser.add_argument('--n_saves', type=int, default=0, metavar='E',
                        help='frequency of weight saves in epochs')
    parser.add_argument('-opt', '--optimizer', type=str,  choices=['SGD', 'Adam'], default='SGD', metavar='LR',
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, metavar='LR',
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0., metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--grad_perturbation', type=float, default=0., metavar='M',
                        help='grad perturbation norm (default: 0.)')
    parser.add_argument('--perturbation_type', type=str, default='RANDOM', choices=['RANDOM', 'FIXED'], metavar='M',
                        help='grad perturbation norm (default: 0.)')

    parser.add_argument('--no_gpu', type=bool, default=False,
                        help='disables GPU training')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('-dp', '--dataset_path', type=str, default='~/data', metavar='P',
                        help='path to dataset (default: /data)')
    parser.add_argument('-sp', '--name', type=str, default='./default_name', metavar='P',
                        help='path to save run')
    return parser
