import os
import numpy as np
import torch
import argparse
from utils import StandardScaler
from MDGCRNAdj import MDGCRNAdj
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import load_adj

def get_model():  
    adj_mx = load_adj(adj_mx_path, args.adj_type)
    adjs = [torch.tensor(i).to(device) for i in adj_mx] 
    model = MDGCRNAdj(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                 rnn_units=args.rnn_units, rnn_layers=args.rnn_layers, embed_dim=args.embed_dim, cheb_k = args.max_diffusion_step, 
                 adj_mx = adjs, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning, 
                 contra_type=args.contra_type, device=device).to(device)
    return model

def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x0 = x[..., 0:1]
    x1 = x[..., 1:2]
    y0 = y[..., 0:1]
    y1 = y[..., 1:2]
    return x0.to(device), x1.to(device), y0.to(device), y1.to(device) # x, x_cov, y, y_cov

def plot_embedding(data, k, path, mode='train'):
    data = data.cpu().detach().numpy()
    fig, axes = plt.subplots(1, k, figsize=(16, 4))
    for i in range(k):
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        embedded_data = tsne.fit_transform(data[i])
        
        ax = axes[i]
        ax.scatter(embedded_data[:, 0], embedded_data[:, 1], c='b', marker='o', alpha=0.5)
        ax.set_title(f'{mode}_subplot {i + 1}')

    plt.tight_layout()
    if args.schema == 0:
        fname = f'{path}/{mode}_samples.png'
    else:
        fname = f'{path}/{mode}_samples.png'
    plt.savefig(fname)
    plt.show()
    return

def visualization(modelpt_path, path):
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    k = 4
    with torch.no_grad():
        model = model.eval()
        for mode in ['train', 'val', 'test']:
            data_iter =  data[f'{mode}_loader']
            for i, (x, y) in enumerate(data_iter):
                if i > 0:  # Only one batch
                    break
                x, x_cov, y, y_cov = prepare_x_y(x, y)
                node_embeddings = model(x, x_cov, y_cov)  # (B, N, e)
            plot_embedding(node_embeddings, k, path, mode)
    return 

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--embed_dim', type=int, default=10, help='embedding dimension for adaptive graph')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=128, help='number of rnn units')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--adj_type", type=str, default='symadj', help="scalap, normlap, symadj, transition")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed.')
# TODO: support contra learning
parser.add_argument('--temp', type=float, default=0.1, help='temperature parameter')
parser.add_argument('--lam', type=float, default=0.1, help='contrastive loss lambda') 
parser.add_argument('--lam1', type=float, default=0.1, help='compact loss lambda') 
parser.add_argument('--schema', type=int, default=1, choices=[0, 1], help='which contra backbone schema to use (0 is no contrast, i.e., baseline)')
parser.add_argument('--contra_type', type=eval, choices=[True, False], default='True', help='whether to use InfoNCE loss or Triplet loss')
args = parser.parse_args()
        
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx.pkl'
    args.num_nodes = 207
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx_bay.pkl'
    args.num_nodes = 325
else:
    pass # including more datasets in the future     

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'../{args.dataset}', category + 'his.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    # data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

data['train_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train']), torch.FloatTensor(data['y_train'])),
    batch_size=args.batch_size,
    shuffle=False
)
data['val_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_val']), torch.FloatTensor(data['y_val'])),
    batch_size=args.batch_size, 
    shuffle=False
)
data['test_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_test']), torch.FloatTensor(data['y_test'])),
    batch_size=args.batch_size, 
    shuffle=False
)

def main():
    model_name = 'MDGCRNAdj'
    if args.schema == 0:
        timestring = '20231110161240'
    else:
        if args.contra_type:
            timestring = '20231110161719'
        else:
            timestring = '20231110161828'
    path = f'../save/{args.dataset}_{model_name}_{timestring}' + '_baseline' if args.schema == 0 else f'../save/{args.dataset}_{model_name}_{timestring}'
    modelpt_path = f'{path}/{model_name}_{timestring}.pt'
    visualization(modelpt_path, path)
    
if __name__ == '__main__':
    main()
    
