import torch

def MSE(y_true, y_pred):
    
    mse = torch.square(y_pred - y_true)
    mse = torch.mean(mse)
    return mse
    
def RMSE(y_true, y_pred):
    
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = torch.sqrt(MSE(y_true, y_pred))
    return rmse
        
def MAE(y_true, y_pred, mask=None):
    
    mae = torch.abs(y_pred - y_true)
    if mask is not None:
        mae_ig = mae * (1 - mask)  # max: 0.5 (0.9422?) min:0, but max should be close 0
        mae = mae * mask  # max: 0.65, min:0
    mae = torch.mean(mae)
    anomoly_num = torch.sum(mask)
    normal_num = torch.sum(1 - mask)  # in general, normal = 3 * anomoly
    mae_ig = torch.mean(mae_ig)  # mae_ig = 3 * mae alought max:0.9422, mean:0.04, i.e., most is accurately predicted
    return mae





