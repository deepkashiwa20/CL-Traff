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
        mae = mae * mask
    mae = torch.mean(mae)
    return mae





