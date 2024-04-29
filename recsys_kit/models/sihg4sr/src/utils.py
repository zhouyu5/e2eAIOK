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

def save_predictions(preds, save_path, file_tail=""):
    import csv
    with open(f'{save_path}/prediction{file_tail}.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["session_id", "item_id", "rank"])
        for sid, item_list in preds:
            for idx, iid in enumerate(item_list):
                spamwriter.writerow([sid, iid, idx + 1])
    print(f"{save_path}/prediction{file_tail}.csv is saved")

def save_logits(logits_to_save, save_path):
    import pandas as pd
    if len(logits_to_save[0]) == 2:
        pdf = pd.DataFrame(logits_to_save, columns =['session_id', 'logits'])
        pdf.to_parquet(f"{save_path}/logits.parquet")
    elif len(logits_to_save[0]) == 3:
        pdf = pd.DataFrame(logits_to_save, columns =['session_id', 'logits', 'labels'])
        if pdf.shape[0] > 40000:
            pdf[:40000].to_parquet(f"{save_path}/logits1.parquet")
            pdf[40000:].to_parquet(f"{save_path}/logits2.parquet")
    print(f"{save_path}/logits.parquet is saved")

def do_append_similiar_item(preds, feature_model, feature_table, neigh):
    append_pres = [[], [], [], []]
    input = feature_table[preds]
    if feature_model:
        input_tensor = th.LongTensor(input)
        logits = feature_model(input_tensor)
        _, topk = logits.topk(4)
        topk = topk.detach().tolist()
    elif neigh:
        _, topk = neigh.kneighbors(input, n_neighbors=5)
    for pred, iids in zip(preds, topk):
        for i in range(4):
            append_pres[i].append(iids[i+1])
    return preds + append_pres[0] + append_pres[1] + append_pres[2] + append_pres[3]

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

def prepare_candidate_filter(candidate_df, num_items, batch_size):
    candidate_list = candidate_df['item_id'].tolist()
    logit_difference = [100 if i in candidate_list else 0 for i in range(num_items)]
    logit_difference_batch = np.array([logit_difference for i in range(batch_size)])
    return logit_difference_batch