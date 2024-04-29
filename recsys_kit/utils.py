import timeit
import sklearn.metrics
import numpy as np
import os
import pandas as pd
import os
from tqdm import tqdm
import pickle as pkl

class Timer:
    level = 0
    viewer = None
    def __init__(self, name):
        self.name = name
        if Timer.viewer:
            Timer.viewer.display(f"{name} started ...")
        else:
            print(f"{name} started ...")

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if Timer.viewer:
            Timer.viewer.display(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        else:
            print(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def fix_na(df):
    df = df.fillna(0)
    for col in df.select_dtypes([float]):
        v = col
        if np.array_equal(df[v], df[v].astype(int)):
            df[v] = df[v].astype(int, copy = False)
    return df

def load_file(file_path):
    if not isinstance(file_path, str):
        file_path = str(file_path)
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        data = pd.read_parquet(file_path)
    elif file_path.endswith(".pkl") or file_path.endswith(".txt"):
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
    else:
        raise NotImplementedError(f"Unable to load {file_path}")    
    return data
 
def load_csv_to_pandasdf(dataset):
    if not isinstance(dataset, str):
        raise NotImplementedError("Only support pandas Dataframe as input")
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"{dataset} is not exists")
    if os.path.isdir(dataset):
        input_files = sorted(os.listdir(dataset))
        df = pd.read_csv(dataset + "/" + input_files[0], sep = '\t')
        for file in tqdm(input_files[1:], desc='load_csv_to_pandasdf'):
            part = pd.read_csv(dataset + "/" + file, sep = '\t')    
            df = pd.concat([df, part],axis=0)
    else:
        df = pd.read_csv(dataset, sep = '\t')
    df = fix_na(df)
    return df

def H_np(y, p):
    e = np.finfo(float).eps
    return -y * np.log(p + e) - (1 - y) * np.log(1 - p + e)
    
def nce_score(y_true, y_pred, verbose = False):
    avg_logloss_y_p = np.mean(sklearn.metrics.log_loss(y_true, y_pred))
    #avg_log_reci_p = np.mean(np.log(1/y_pred))
    ctr = y_true.sum() / y_true.shape[0]
    if not verbose:
        logloss_ctr = H_np(ctr, ctr)
        return avg_logloss_y_p / logloss_ctr

def get_combined_df(file_list, save_path=None, weights = []):
    model_num = len(file_list)
    if len(weights) == 0:
        weights = [1/model_num] * model_num
    df = pd.read_csv(file_list[0], sep='\t')\
        .rename(columns={"row_id": "RowId"})
    df['is_installed'] = df['is_installed'] * weights[0]
    df_seq = df[['RowId']]
    for i in range(1, model_num):
        df_temp = pd.read_csv(file_list[i], sep='\t')\
            .rename(columns={"row_id": "RowId"})
        df_temp = df_seq.merge(df_temp, on='RowId', how='left').reset_index(drop=True)
        df['is_installed'] += df_temp['is_installed'] * weights[i]
    df['is_clicked'] = 0.0
    if save_path:
        df.round({'is_clicked': 1, 'is_installed': 5})\
            .to_csv(save_path, sep='\t', header=True, index=False)
    return df


def init_accelerator(device='cpu'):
    import torch
    if device == 'hpu':
        import habana_frameworks.torch.core as htcore
        device = torch.device("hpu:0")
    elif device == 'xpu':
        import intel_extension_for_pytorch as ipex
        device = torch.device("xpu:0")
    elif device == 'cuda':
        device = torch.device("cuda:0")
    elif device == 'cpu':
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device type {device}")\
        
    torch.set_default_device(device)
        
    return device