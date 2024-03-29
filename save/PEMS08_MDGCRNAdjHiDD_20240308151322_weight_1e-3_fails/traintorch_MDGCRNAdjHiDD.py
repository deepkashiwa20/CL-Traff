import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchinfo import summary
import argparse
import logging
from utils import StandardScaler, masked_mae_loss, masked_mape_loss, masked_mse_loss, masked_rmse_loss
from utils import load_adj
from metrics import RMSE, MAE, MSE
from MDGCRNAdjHiDD import MDGCRNAdjHiDD

class ContrastiveLoss():
    def __init__(self, contra_loss='triplet', mask=None, temp=1.0, margin=1.0):
        self.infonce = contra_loss in ['infonce']
        self.mask = mask
        self.temp = temp
        self.margin = margin
    
    def calculate(self, query, pos, neg, mask):
        """
        :param query: shape (batch_size, num_sensor, hidden_dim)
        :param pos: shape (batch_size, num_sensor, hidden_dim)
        :param neg: shape (batch_size, num_sensor, hidden_dim) or (batch_size, num_sensor, num_memory, hidden_dim)
        :param mask: shape (batch_size, num_sensor, num_memory) True means positives
        """
        if not self.infonce:
            separate_loss = nn.TripletMarginLoss(margin=self.margin)
            return separate_loss(query, pos.detach(), neg.detach())
        else:
            score_matrix = F.cosine_similarity(query.unsqueeze(-2), neg, dim=-1)  # (B, N, M)
            score_matrix = torch.exp(score_matrix / self.temp)
            pos_sum = torch.sum(score_matrix * mask, dim=-1)
            ratio = pos_sum / torch.sum(score_matrix, dim=-1)
            u_loss = torch.mean(-torch.log(ratio))
            return u_loss  

def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return

def get_model():
    adj_mx = load_adj(adj_mx_path, args.adj_type)
    adjs = [torch.tensor(i).to(device) for i in adj_mx]            
    model = MDGCRNAdjHiDD(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                 rnn_units=args.rnn_units, rnn_layers=args.rnn_layers, cheb_k = args.max_diffusion_step, mem_num=args.mem_num, 
                 mem_dim=args.mem_dim, embed_dim=args.embed_dim, adj_mx = adjs, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning, 
                 contra_loss=args.contra_loss, diff_max=diff_max, diff_min=diff_min, use_mask=args.use_mask, use_STE=args.use_STE, device=device).to(device)
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
    x2 = x[..., 2:3]  
    y0 = y[..., 0:1]
    y1 = y[..., 1:2]
    y2 = y[..., 2:3]
    return x0.to(device), x1.to(device), x2.to(device), y0.to(device), y1.to(device), y2.to(device) # x, x_cov, y, y_cov

def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  data[f'{mode}_loader']
        ys_true, ys_pred = [], []
        losses = []
        for x, y in data_iter:
            x, x_cov, x_his, y, y_cov, _ = prepare_x_y(x, y)
            output, h_att, query, pos, neg, mask, real_dis, latent_dis, mask_dis = model(x, x_cov, scaler.transform(x_his), y_cov)
            y_pred = scaler.inverse_transform(output)
            y_true = y
            ys_true.append(y_true)
            ys_pred.append(y_pred)
            losses.append(masked_mae_loss(y_pred, y_true).item())
        
        ys_true, ys_pred = torch.cat(ys_true, dim=0), torch.cat(ys_pred, dim=0)
        loss = masked_mae_loss(ys_pred, ys_true)

        if mode == 'test':
            mae = masked_mae_loss(ys_pred, ys_true).item()
            mape = masked_mape_loss(ys_pred, ys_true).item()
            rmse = masked_rmse_loss(ys_pred, ys_true).item()
            mae_3 = masked_mae_loss(ys_pred[:, 2, ...], ys_true[:, 2, ...]).item()
            mape_3 = masked_mape_loss(ys_pred[:, 2, ...], ys_true[:, 2, ...]).item()
            rmse_3 = masked_rmse_loss(ys_pred[:, 2, ...], ys_true[:, 2, ...]).item()
            mae_6 = masked_mae_loss(ys_pred[:, 5, ...], ys_true[:, 5, ...]).item()
            mape_6 = masked_mape_loss(ys_pred[:, 5, ...], ys_true[:, 5, ...]).item()
            rmse_6 = masked_rmse_loss(ys_pred[:, 5, ...], ys_true[:, 5, ...]).item()
            mae_12 = masked_mae_loss(ys_pred[:, 11, ...], ys_true[:, 11, ...]).item()
            mape_12 = masked_mape_loss(ys_pred[:, 11, ...], ys_true[:, 11, ...]).item()
            rmse_12 = masked_rmse_loss(ys_pred[:, 11, ...], ys_true[:, 11, ...]).item()
            
            logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae, mape * 100, rmse))
            logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_3, mape_3 * 100, rmse_3))
            logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_6, mape_6 * 100, rmse_6))
            logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_12, mape_12 * 100, rmse_12))

        return np.mean(losses), ys_true, ys_pred

    
def traintest_model():  
    model = get_model()
    print_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader']
        losses, mae_losses, contra_losses, compact_losses, detect_losses = [], [], [], [], []
        for x, y in data_iter:
            optimizer.zero_grad()
            x, x_cov, x_his, y, y_cov, _ = prepare_x_y(x, y)
            output, h_att, query, pos, neg, mask, real_dis, latent_dis, mask_dis = model(x, x_cov, scaler.transform(x_his), y_cov, scaler.transform(y), batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = y
            mae_loss = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            separate_loss = ContrastiveLoss(contra_loss=args.contra_loss, mask=mask, temp=args.temp)
            u_loss = separate_loss.calculate(query, pos, neg, mask)
            if args.compact_loss == 'mse':
                compact_loss = nn.MSELoss()
            elif args.compact_loss == 'rmse':
                compact_loss = RMSE
            elif args.compact_loss == 'mae':
                compact_loss = MAE
            else:
                pass
            loss_c = compact_loss(query, pos.detach())
            
            if args.detect_loss == 'mse':
                detect_loss = nn.MSELoss()
            elif args.detect_loss == 'rmse':
                detect_loss = RMSE
            elif args.detect_loss == 'mae':
                detect_loss = nn.L1Loss()
            else:
                pass
            # loss_d = detect_loss(real_dis, latent_dis, mask=mask_d)
            loss_d = detect_loss(real_dis, latent_dis)
            loss = mae_loss + args.lamb * u_loss + args.lamb1 * loss_c + args.lamb2 * loss_d
            losses.append(loss.item())
            mae_losses.append(mae_loss.item())
            contra_losses.append(u_loss.item())
            compact_losses.append(loss_c.item())
            detect_losses.append(loss_d.item())
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
        train_loss = np.mean(losses)
        train_mae_loss = np.mean(mae_losses) 
        train_contra_loss = np.mean(contra_losses)
        train_compact_loss = np.mean(compact_losses)
        train_detect_loss = np.mean(detect_losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, train_mae_loss: {:.4f}, train_contra_loss: {:.4f}, train_compact_loss: {:.4f}, train_detect_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, args.epochs, batches_seen, train_loss, train_mae_loss, train_contra_loss, train_compact_loss, train_detect_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        test_loss, _, _ = evaluate(model, 'test')

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % (epoch_num + 1))
                break
    
    logger.info('=' * 35 + 'Best val_loss model performance' + '=' * 35)
    logger.info('=' * 22 + 'Better results might be found from model at different epoch' + '=' * 22)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, _, _ = evaluate(model, 'test')

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY','PEMS03','PEMS04','PEMS07','PEMS08'], default='METRLA', help='which dataset to run')
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
parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=30, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--adj_type", type=str, default='symadj', help="scalap, normlap, symadj, transition, doubletransition")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed.')
# TODO: support contra learning
parser.add_argument('--temp', type=float, default=1.0, help='temperature parameter')
parser.add_argument('--lamb', type=float, default=0.1, help='contra loss lambda') 
parser.add_argument('--lamb1', type=float, default=0.0, help='compact loss lambda') 
parser.add_argument('--lamb2', type=float, default=1.0, help='anomaly detection loss lambda') 
parser.add_argument('--contra_loss', type=str, choices=['triplet', 'infonce'], default='infonce', help='whether to triplet or infonce contra loss')
parser.add_argument('--compact_loss', type=str, choices=['mse', 'rmse', 'mae'], default='mse', help='which method to calculate compact loss')
parser.add_argument('--detect_loss', type=str, choices=['mse', 'rmse', 'mae'], default='mae', help='which method to calculate detect loss')
parser.add_argument("--use_mask", type=eval, choices=[True, False], default='False', help="use mask to calculate detect loss")
parser.add_argument("--use_STE", type=eval, choices=[True, False], default='True', help="use spatio-temporal embedding")
args = parser.parse_args()
num_nodes_dict={
    "PEMS03": 358,
    "PEMS04": 307,
    "PEMS07": 883,
    "PEMS08": 170,
}
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx.pkl'
    args.num_nodes = 207
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx_bay.pkl'
    args.num_nodes = 325
    args.cl_decay_steps = 8000
    args.steps = [10, 150]
else:
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    # args.cl_decay_steps = 8000
    # args.val_ratio=0.25
    # args.steps = [10, 150]
    args.steps = [100]

model_name = 'MDGCRNAdjHiDD'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)
message = ''.join([f'{k}: {v}\n' for k, v in vars(args).items()])
logger.info(message)

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
    data['x_' + category] = np.nan_to_num(cat_data['x']) if True in np.isnan(cat_data['x']) else cat_data['x']
    data['y_' + category] = np.nan_to_num(cat_data['y']) if True in np.isnan(cat_data['y']) else cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

#* 既然max都相同, min干脆设置为0, 因为abs的最小值必定>=0, 这样同样能合理解释, 也能归一化到[0, 1],只不过最小值为0.07左右, 与0接近
diff_max = np.max(np.abs(scaler.transform(data['x_train'][..., 0]) - scaler.transform(data['x_train'][..., -1])))  # 3.734067777528973 for x_train, x_val, and x_test
# diff_min = np.min(np.abs(scaler.transform(data['x_train'][..., 0]) - scaler.transform(data['x_train'][..., -1])))  # x_train: 0.34289610787771085, x_val: 0.37793285841246993, x_test: 0.2914432740946036
diff_min = 0.

data['train_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train']), torch.FloatTensor(data['y_train'])),
    batch_size=args.batch_size,
    shuffle=True
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
    logger.info(args.dataset, 'training and testing started', time.ctime())
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()
    
# nohup python traintest_DGCRN.py --gpu 3 > LA_noTrY.log 2>&1 &
