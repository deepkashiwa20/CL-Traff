import torch

def MSE(y_true, y_pred):
    
    mse = torch.square(y_pred - y_true)
    mse = torch.mean(mse)
    return mse
    
def RMSE(y_true, y_pred):
    
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = torch.sqrt(MSE(y_true, y_pred))
    return rmse
        
def MAE(y_true, y_pred):
    
    mae = torch.abs(y_pred - y_true)
    mae = torch.mean(mae)
    return mae





