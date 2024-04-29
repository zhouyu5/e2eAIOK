import os
import pickle
import argparse
import pandas as pd

from recsys_kit.utils import Timer, load_file
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile

import sys
sys.path.append('/src')

def format_list(l):
    return l.tolist() if not isinstance(l, list) else l

def add_extra(processed, num_unique_features, extra_feat_key, item_features_extra_df_orig):
    if extra_feat_key is None or len(extra_feat_key) == 0:
        return processed, num_unique_features
    print(f"Start to add {extra_feat_key} to original 904 features")
    processed.reset_index(drop=True, inplace=True)
    item_features_extra_df = item_features_extra_df_orig.copy()
    num_feats = num_unique_features
    for feat_name in extra_feat_key:
        # because minimun value for one feature can be -1, add num_feats with 1 firstly
        num_feats += 1 #905
        # [ ...... ]904 + [-1, 0, 1, 2, 3] 909
        item_features_extra_df[feat_name] = item_features_extra_df[feat_name] + num_feats
        default = num_feats - 1
        print(f"num_feats is {num_feats}")
        num_feats = (item_features_extra_df[feat_name].max() + 1)
        print(f"after add extra, num_feats is {num_feats}")
        item_feat_dict = dict((iid, fid) for iid, fid in zip(item_features_extra_df['item_id'].to_list(), item_features_extra_df[feat_name].to_list()))
        new_feature = []
        for k, x in tqdm(zip(processed['item_id'].to_list(), processed['feature'].to_list()), total = len(processed["feature"])):
            k = format_list(k)
            x = format_list(x)
            assert(len(k) == len(x))
            new_feature.append([format_list(fl) + [item_feat_dict[iid] if iid in item_feat_dict else default] for iid, fl in zip(k, x)])
        processed['feature'] = pd.Series(new_feature)
    num_unique_features = num_feats
    return processed, num_unique_features

def add_sesstime(processed, num_unique_features, extra_feat_key):
    if extra_feat_key is None or len(extra_feat_key) == 0:
        return processed, num_unique_features
    print(f"Start to add {extra_feat_key} to original 904 features")
    processed.reset_index(drop=True, inplace=True)
    num_feats = num_unique_features
    for feat_name in extra_feat_key:
        new_max = num_feats + 4
        # because minimun value for one feature can be -1, add num_feats with 1 firstly
        # [ ...... ]904 + [0, 1, 2, 3] 909
        print(f"num_feats is {num_feats}")
        new_feature = []
        for x, fl in tqdm(zip(processed['feature'].to_list(), processed[feat_name].to_list()), total = len(processed["feature"])):
            x = format_list(x)
            fl = [f + num_feats for f in fl]
            new_feature.append([format_list(orig_fl) + [f] for orig_fl, f in zip(x, fl)])
        processed['feature'] = pd.Series(new_feature)
        num_feats = new_max
        print(f"after add extra, num_feats is {num_feats}")
    num_unique_features = num_feats
    return processed, num_unique_features

def exclude_feat(processed, exclude_feat_ids):
    if exclude_feat_ids is None or len(exclude_feat_ids) == 0:
        return processed
    if len(exclude_feat_ids) == 0:
        return processed
    print(f"Start to exclude feature {exclude_feat_ids} in original 904 features")
    processed.reset_index(drop=True, inplace=True)
    new_feature = []
    new_feature_cat = []
    for k, x in tqdm(zip(processed['feature'].to_list(), processed['feature_cat'].to_list()), total = len(processed["feature"])):
        k = format_list(k)
        x = format_list(x)
        # k is [[feat0, feat8, feat1], [feat2, feat1, feat3], ... ]
        # x is [[cat0, cat1, cat3], [cat3, cat3, cat8], ...]
        assert(len(k) == len(x))
        new_fl = []
        new_cl = []
        for fl, cl in zip(k, x):
            # fl is [feat0, feat8, feat1]
            # cl is [cat0, cat1, cat3]
            fl = format_list(fl)
            cl = format_list(cl)
            new_fl.append([f for f in fl if f not in exclude_feat_ids])
            new_cl.append([c for f, c in zip(fl, cl) if f not in exclude_feat_ids])
        new_feature.append(new_fl)
        new_feature_cat.append(new_cl)
    processed['feature'] = pd.Series(new_feature)
    processed['feature_cat'] = pd.Series(new_feature_cat)
    return processed

class DataPrep:
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.metadata = None

    def prepare_features(self):
        if self.metadata is not None:
            return

        if os.path.exists(self.feat_dir):
            feat_dir = self.feat_dir
            with open(os.path.join(feat_dir, "meta.pkl"), 'rb') as file:
                self.metadata = pickle.load(file)
        self.input_feature_names = ['session_id', 'item_id', 'date']

    def _transform(self, session, target, num_unique_features, kg_feat_dict, kg_feat_cat_dict, start_ts, total_duration, item_features_extra_df):
        if target is not None:
            target = target.rename(columns={'item_id': 'y', 'date': 'purchase_date'})
            target['purchase_date'] = pd.to_datetime(target["purchase_date"], format='mixed')
        session['date'] = pd.to_datetime(session["date"], format='mixed')
        
        with Timer("add elapse to start time and end time feature"):
            grouped = session.groupby('session_id').agg(start_time=('date','min'), end_time=('date','max'))
            session = session.merge(grouped, on='session_id', how='left')
            session['elapse_to_start'] = ((session['date'] - session['start_time']).dt.seconds/60).astype(int)
            session['elapse_to_end'] = ((session['end_time'] - session['date']).dt.seconds/60).astype(int)
            session['binned_elapse_to_start'] = pd.cut(session['elapse_to_start'], [-1, 0, 3, 15, 1434]).cat.codes
            session['binned_elapse_to_end'] = pd.cut(session['elapse_to_end'], [-1, 0, 3, 16, 1434]).cat.codes

        with Timer("combine same session as one record"):
            processed = session.groupby("session_id", as_index = False).agg({'item_id':lambda x: list(x), 'binned_elapse_to_start':lambda x: list(x), 'binned_elapse_to_end':lambda x: list(x),})
            if target is not None:
                processed = target.merge(processed, how="inner", on="session_id")

        with Timer("add features to each item"):
            # map features to processed
            feature_list_series = []
            feature_cat_list_series = []
            for idx, item_id_list in tqdm(processed["item_id"].items(), total = len(processed["item_id"])):
                item_feature_list = []
                item_feature_cat_list = []
                for item_id in item_id_list:
                    # we need to add item feature and other created features
                    item_feature_list.append(kg_feat_dict[item_id])
                    item_feature_cat_list.append(kg_feat_cat_dict[item_id])
                feature_list_series.append(item_feature_list)
                feature_cat_list_series.append(item_feature_cat_list)
            processed["feature"] = pd.Series(feature_list_series)
            processed["feature_cat"] = pd.Series(feature_cat_list_series)
            
        if target is not None:
            with Timer("get the weighted factor for session based on ts"):
                weighted_factor_list_series = []
                for _, ts in tqdm(processed["purchase_date"].items(), total=len(processed["purchase_date"])):
                    weighted_factor_list_series.append((ts - start_ts) / (2 * total_duration) + 0.5)
                processed["wf"] = pd.Series(weighted_factor_list_series)

        # 3.4 include/exclude some features
        if item_features_extra_df is not None:
            processed, num_unique_features = add_extra(processed, num_unique_features, ['binned_count_item_clicks'], item_features_extra_df)
            processed, num_unique_features = add_sesstime(processed, num_unique_features, ['binned_elapse_to_end'])
            processed = exclude_feat(processed, [29, 44])
        
        if target is None:
            processed["wf"] = pd.Series([0] * len(processed))
            processed['y'] = pd.Series([None] * len(processed))
            processed['purchase_date'] = pd.Series([None] * len(processed))
        
        return processed[["item_id", "y", "session_id", "feature", "feature_cat", "wf", "purchase_date"]], num_unique_features
        
    def fit_transform(self, data_path, data_save_dir):
        # 1. prepare output dirs
        data_output_dir = os.path.join(data_save_dir, "processed")
        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir, exist_ok=True)
        
        feat_save_dir = self.feat_dir
        if not os.path.exists(feat_save_dir):
            os.makedirs(feat_save_dir, exist_ok=True)

        # 2. split data
        path = Path(data_path)
        train_target = load_file(path.joinpath("train_purchases.csv"))
        train_session = load_file(path.joinpath("train_sessions.csv"))
        
        valid_target = load_file(path.joinpath("test_leaderboard_purchases.csv"))
        valid_session = load_file(path.joinpath("test_leaderboard_sessions.csv"))
        
        # 3.1 prepare features kg
        with Timer("prepare kg_feat_dict and kg_feat_cat_dict"):
            kg_df = load_file(path.joinpath("item_features.csv"))
            kg_df['feature_category_id'] = kg_df['feature_category_id'].astype("string")
            kg_df['feature_value_id'] = kg_df['feature_value_id'].astype("string")
            kg_df["feature_merge"] = "f_" + kg_df['feature_category_id'] + "=" + kg_df['feature_value_id']
            codes, uniques = pd.factorize(kg_df["feature_merge"])
            # categorify all features in item_features
            kg_df["feature"] = pd.Categorical(codes, categories=range(len(uniques)))
            num_unique_features = len(uniques)
            print(f"num_unique_features is {num_unique_features}")
            kg_feat_dict = dict()
            kg_feat_cat_dict = dict()
            for row in kg_df.to_dict('records'):
                if row['item_id'] not in kg_feat_dict:
                    kg_feat_dict[row['item_id']] = []
                kg_feat_dict[row['item_id']].append(row['feature'])
                if row['item_id'] not in kg_feat_cat_dict:
                    kg_feat_cat_dict[row['item_id']] = []
                kg_feat_cat_dict[row['item_id']].append(int(row['feature_category_id']))
        
        
        # 3.2 prepare item_features_extra_df
        # with Timer("prepare item_clicks features"):
        #     item_clicks = pd.concat([train_session, train_target], ignore_index=True).groupby("item_id", as_index=False).agg(count_item_clicks=('item_id', 'count'))
        #     item_clicks['binned_count_item_clicks'] = pd.cut(item_clicks['count_item_clicks'], [1, 15, 111, 313, 23165]).cat.codes
        #     item_clicks['binned_count_item_clicks'] = item_clicks['binned_count_item_clicks'].astype(int)
        #     item_features_extra_df = item_clicks[['item_id', 'binned_count_item_clicks']]
        item_features_extra_df = None
        
        # 3.3 prepare train_session
        with Timer("prepare duration features"):
            total_duration = pd.to_datetime("2021/06/30") - pd.to_datetime("2020/01/01")
            start_ts = pd.to_datetime("2020/01/01")

        # 4. process train, valid
        save_path = Path(data_output_dir)
        with Timer("process train Data"):
            train_processed, num_unique_features = self._transform(train_session, train_target, num_unique_features, kg_feat_dict, kg_feat_cat_dict, start_ts, total_duration, item_features_extra_df)
            train_processed.to_parquet(save_path.joinpath("train_processed.parquet"), compression = None)
        
        with Timer("process validate Data"):
            valid_processed, num_unique_features = self._transform(valid_session, valid_target, num_unique_features, kg_feat_dict, kg_feat_cat_dict, start_ts, total_duration, item_features_extra_df)
            valid_processed.to_parquet(save_path.joinpath("valid_processed.parquet"), compression = None)

        # 4. Save to disk        
        meta = {
            'num_items': 28144,
            "num_unique_features": num_unique_features,
            "kg_feat_dict": kg_feat_dict, 
            "kg_feat_cat_dict": kg_feat_cat_dict,
            "item_features_extra_df": item_features_extra_df,
            "total_duration": total_duration,
            "start_ts": start_ts,
            'time_feature': "purchase_date",
            'feature_list': ["item_id", "y", "session_id", "feature", "feature_cat", "wf"],
        }
        with open(os.path.join(feat_save_dir, "meta.pkl"), 'wb') as file:
            pickle.dump(meta, file)
            
        copyfile(path.joinpath("candidate_items.csv"), save_path.joinpath("candidate_items.csv"))

    def augement_data(self, data_path):
        path = Path(data_path)
        processed = load_file(path.joinpath("train_processed.parquet"))

        with Timer("expand current data with clicks, may take couple of minutes ..."):
            concat_list = [processed]
            aug_processed = pd.concat(concat_list).reset_index(drop=True)
            to_zip = [aug_processed['item_id'].to_list(), aug_processed['y'].to_list()]

            item_id_list = []
            for item_id, y in tqdm(zip(*to_zip), total = len(aug_processed['item_id'])):
                if not isinstance(item_id, list):
                    item_id = item_id.tolist()
                item_id_new = item_id + [y]
                len_item_id_new = len(item_id_new)
                item_id_new = [item_id_new[:num_item] for num_item in range(2, len_item_id_new + 1)]
                item_id_list.append(item_id_new)
            aug_processed['item_id'] = pd.Series(item_id_list)
            aug_processed = aug_processed.explode('item_id').dropna(subset=['item_id']).reset_index(drop=True)

            item_id_list = []
            y_list = []
            feature_list = []
            feature_cat_list = []
            to_zip = [aug_processed['item_id'].to_list(), aug_processed['feature'].to_list(), aug_processed['feature_cat'].to_list()]
            for item_id_new, feature, feature_cat in tqdm(zip(*to_zip), total = len(aug_processed['item_id'])):
                if len(item_id_new) >= 2:
                    item_id_list.append(item_id_new[:-1])
                    y_list.append(item_id_new[-1])
                else:
                    item_id_list.append(None)
                    y_list.append(None)
                feature_list.append(feature[:len(item_id_new[:-1])])
                feature_cat_list.append(feature_cat[:len(item_id_new[:-1])])

            aug_processed['item_id'] = pd.Series(item_id_list)
            aug_processed['y'] = pd.Series(y_list)
            aug_processed['feature'] = pd.Series(feature_list)
            aug_processed['feature_cat'] = pd.Series(feature_cat_list)

            processed = aug_processed[["item_id", "y", "session_id", "feature", "feature_cat", "wf"]].reset_index(drop=True)
            processed.to_parquet(path.joinpath("train_processed_augmented_with_clicks.parquet"))
        return processed

    def transform(self, data):
        assert isinstance(data, pd.DataFrame)
        test_df = data
        
        # 2. prepare meta
        num_unique_features = self.metadata['num_unique_features']
        kg_feat_dict = self.metadata['kg_feat_dict']
        kg_feat_cat_dict = self.metadata['kg_feat_cat_dict']
        start_ts = self.metadata['start_ts']
        total_duration = self.metadata['total_duration']
        item_features_extra_df = self.metadata['item_features_extra_df']
        
        # 3. actual process
        test_df, _ = self._transform(
            session=test_df, 
            target=None, 
            num_unique_features=num_unique_features, 
            kg_feat_dict=kg_feat_dict, 
            kg_feat_cat_dict=kg_feat_cat_dict, 
            start_ts=start_ts, 
            total_duration=total_duration, 
            item_features_extra_df=item_features_extra_df)
        print(test_df)
        test_df.to_parquet(os.path.join(self.feat_dir, "tmp.parquet"))
        return test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="/dataset", type=str)
    parser.add_argument("--feat_dir", dest="feat_dir", default="/feature_store", type=str)
    parser.add_argument("--save_dir", dest="save_dir", default="/output", type=str)
    parser.add_argument("--train", dest="is_train", action="store_true", default=False)
    parser.add_argument("--predict", dest="is_predict", action="store_true", default=False)
    parser.add_argument("--augment", dest="is_augment", action="store_true", default=False)
    args = parser.parse_args()
    
    dataprep = DataPrep(feat_dir = args.feat_dir)
    
    if args.is_train:
        dataprep.fit_transform(data_path = args.data_dir, data_save_dir=args.save_dir)

    elif args.is_predict:
        test_df = load_file(os.path.join(args.data_dir, "test.csv")).reset_index(drop=True)
        print(test_df.dtypes)
        dataprep.prepare_features()
        processed_test_df = dataprep.transform(test_df)
        print(processed_test_df)
        processed_test_df.to_parquet(os.path.join(args.save_dir, "processed", "test_processed.parquet"))
        
    elif args.is_augment:
        dataprep.augement_data(data_path = args.data_dir)

    else:  
        raise ValueError(f'Please either use --train or --predict flag')
