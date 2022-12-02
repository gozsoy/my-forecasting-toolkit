import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def process_series_tcn(df, input_size, output_size, stride):

    X = np.empty(shape=(1,input_size,1))
    y = np.empty(shape=(1,input_size))

    for idx, _ in df.iterrows():
        
        if idx % stride == 0 and idx+input_size+output_size <= len(df):
            
            input_values = (df.iloc[idx:idx+input_size]).to_numpy()
            input_values = np.expand_dims(input_values,0)

            output_values = (df.iloc[idx+output_size:idx+input_size+output_size]).to_numpy().T
            
            X = np.concatenate((X,input_values))
            y = np.concatenate((y,output_values))
    
    return X[1:],y[1:]


def process_series_nbeats(df, input_size, output_size, stride):

    X = np.empty(shape=(1,input_size))
    y = np.empty(shape=(1,output_size))

    for idx, _ in df.iterrows():
        
        if idx % stride == 0 and idx+input_size+output_size <= len(df):
            
            input_values = (df.iloc[idx:idx+input_size]).to_numpy().T

            output_values = (df.iloc[idx+input_size:idx+input_size+output_size]).to_numpy().T
            
            X = np.concatenate((X,input_values))
            y = np.concatenate((y,output_values))
    
    return X[1:],y[1:]


def rmse(y_true, y_pred):
    return mean_squared_error(y_true,y_pred,squared=False)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)