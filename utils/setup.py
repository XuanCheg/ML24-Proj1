from model.KNN import KNN
from model.MLP import MLP
from model.TCN import TemporalConvNet
from model.SVM import SVM
from torch import nn


def setup_model(opt, input_dim, output_dim, device='cuda'):
    if opt.model == 'MLP':
        model = MLP(input_dim, opt.hdim, output_dim, opt.num_layers, dropout=opt.dropout)
    elif opt.model == 'TCN':
        # FOR ADNI
        tcn = TemporalConvNet(input_dim[0], [128, 48, 16, 1], opt.kernel_size, opt.dropout)
        mlp = MLP(input_dim[1], opt.hdim, output_dim, opt.num_layers, dropout=opt.dropout)
        model = nn.Sequential(tcn, mlp)
    elif opt.model == 'KNN':
        model = KNN(n_neighbors=opt.n_neighbors)
    elif opt.model == 'SVM':
        model = SVM(random_state=opt.seed)
    else:
        raise ValueError("Unknown model")

    return model.to(device)