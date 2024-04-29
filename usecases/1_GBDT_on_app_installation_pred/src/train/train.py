import os
import lightgbm as lgb
import argparse
import pandas as pd
import pickle

import sys
sys.path.append("/src")

from utils import Timer

def main(feat_dir, data_dir):
    num_trees = 10000
    metric_freq = 1000
    target_label = 'is_installed'
    
    data_path = os.path.join(data_dir, "processed", "train_processed.parquet")
    train_df = pd.read_parquet(data_path).reset_index(drop=True)
    
    model_dir = os.path.join(data_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lgbm_trained_HPO1.mdl")

    with open(os.path.join(feat_dir, "meta.pkl"), 'rb') as file:
        metadata = pickle.load(file)
    selected_features = metadata['output_feature_names']
    excluded_features = ['f_0', 'f_7', 'f_1', "is_clicked", "is_installed"]
    features_train = [i for i in selected_features if i not in excluded_features and i in train_df.columns]

    print(" ########################## begin training ##########################", flush="True")

    lgbm_parms = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss'],
        'num_leaves': 63,
        'max_bin': 255,
        'num_trees': num_trees,
        'min_data_in_leaf': 20,
        'min_sum_hessian_in_leaf': 5.0,
        'is_enable_sparse': True,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }

    with Timer(f"Training"):
        dtrain = lgb.Dataset(
            data=train_df[features_train], 
            label=train_df[target_label]
        )

        model = lgb.train(
            lgbm_parms,
            train_set=dtrain,
            valid_sets=dtrain,
            callbacks=[lgb.log_evaluation(metric_freq)],
        )

    model.save_model(model_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="/output", type=str)
    parser.add_argument("--feat_dir", dest="feat_dir", default="/feature_store", type=str)
    args = parser.parse_args()
    
    main(args.feat_dir, args.data_dir)