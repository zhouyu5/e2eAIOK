import os
import pickle
import argparse
import pandas as pd

import sys
sys.path.append('/src')

from utils import load_csv_to_pandasdf
from fe_utils import NewValueEncoder, CountEncoder, Indexer, fit_count_encoded_feature, fit_indexing_feature, fit_newvalue, transform_newvalue, transform_count_encoded_feature, transform_indexing_feature

class DataPrep:
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.metadata = None
        self.new_value_encoder_list = None
        self.count_encoder_list = None
        self.index_encoder_list = None
        self.input_feature_names = None
        self.output_feature_names = None
        
    def prepare_features(self):
        if self.metadata is not None:
            return

        if os.path.exists(self.feat_dir):
            feat_dir = self.feat_dir
            with open(os.path.join(feat_dir, "meta.pkl"), 'rb') as file:
                self.metadata = pickle.load(file)
            
            self.input_feature_names = self.metadata['input_feature_names']
            self.output_feature_names = self.metadata['output_feature_names']
            excluded_features = ['f_0', 'f_7', 'f_1', "is_clicked", "is_installed"]
            self.output_feature_names = [i for i in self.output_feature_names if i not in excluded_features]
            
            fdflag_features = self.metadata['fdflag_features']
            count_features = self.metadata['count_features']
            index_features = self.metadata['index_features']

            new_value_encoder_list = {}
            count_encoder_list = {}
            index_encoder_list = {}

            for feature_name in fdflag_features:
                new_value_encoder_list[feature_name] = NewValueEncoder.load_model(os.path.join(feat_dir,  f"newvalue_{feature_name}.pkl"))

            for feature_name in count_features:
                count_encoder_list[feature_name] = CountEncoder.load_model(os.path.join(feat_dir,  f"count_{feature_name}.pkl"))

            for feature_name in index_features:
                index_encoder_list[feature_name] = Indexer.load_model(os.path.join(feat_dir,  f"index_{feature_name}.pkl"))
                
            self.new_value_encoder_list = new_value_encoder_list
            self.count_encoder_list = count_encoder_list
            self.index_encoder_list = index_encoder_list
    
    def fit_transform(self, data_path, data_save_dir):
        # 1. prepare output dirs
        feat_save_dir = self.feat_dir
        valid_data_dir = os.path.join(data_path, "valid")
        train_data_dir = os.path.join(data_path, "train")
        os.makedirs(valid_data_dir, exist_ok=True)
        
        data_output_dir = os.path.join(data_save_dir, "processed")
        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir, exist_ok=True)

        # 2. train/valid split
        valid_date = 66
        time_feature = 'f_1'

        train_df_origin = load_csv_to_pandasdf(train_data_dir)
        train_df_origin = train_df_origin.sort_values(by=[time_feature]).reset_index(drop=True)

        train_df = train_df_origin[train_df_origin[time_feature] < valid_date].copy()
        valid_df = train_df_origin[train_df_origin[time_feature] == valid_date].copy()

        valid_df.to_csv(os.path.join(valid_data_dir, "valid.csv"), index=False, sep=',')

        # 3. actual data process
        input_feature_names = [f'f_{i}' for i in range(80)] + ["is_clicked", "is_installed"]
        fdflag_features = [f"f_{i}" for i in list(range(2, 23)) + [78, 75, 50]]
        count_features = [f"f_{i}" for i in [2, 4, 6, 13, 15, 18] + [78, 75, 50, 20, 24]]
        index_features = [f"f_{i}" for i in list(range(2, 23))]
        selected_features = ['dow']
        selected_features += [f"f_{i}" for i in range(0, 80)]
        selected_features += [f"f_{i}_CE" for i in [2, 4, 6, 13, 15, 18]+[78, 75, 50, 20, 24]]
        selected_features += [f"f_{i}_idx" for i in range(2, 23) if i not in [2, 4, 6, 15]]
        output_feature_names = selected_features

        partition_key = 'f_35'

        train_df['dow'] = train_df['f_1'] % 7

        train_df, new_value_encoder_list = fit_newvalue(fdflag_features, train_df)
        train_df, count_encoder_list = fit_count_encoded_feature(count_features, train_df)
        train_df, index_encoder_list = fit_indexing_feature(index_features, partition_key, train_df)

        # 4. Save to disk
        train_df.to_parquet(os.path.join(data_output_dir, "train_processed.parquet"))
        
        self.metadata = {
            "fdflag_features": fdflag_features,
            "count_features": count_features,
            "index_features": index_features,
            "input_feature_names": input_feature_names,
            "output_feature_names": output_feature_names,
        }
        with open(os.path.join(feat_save_dir, "meta.pkl"), 'wb') as file:
            pickle.dump(self.metadata, file)
        
        for feature_name, encoder in new_value_encoder_list.items():
            encoder.save(os.path.join(feat_save_dir,  f"newvalue_{feature_name}.pkl"))
        for feature_name, encoder in count_encoder_list.items():
            encoder.save(os.path.join(feat_save_dir,  f"count_{feature_name}.pkl"))
        for feature_name, encoder in index_encoder_list.items():
            encoder.save(os.path.join(feat_save_dir,  f"index_{feature_name}.pkl"))

    def transform(self, data):
        enable_numpy = False
        # 1. prepare data
        if isinstance(data, pd.DataFrame):
            test_df = data
        else:
            test_df = pd.DataFrame(data, columns = self.input_feature_names)
            enable_numpy = True
        
        # 2. actual process
        fdflag_features = self.new_value_encoder_list.keys()
        count_features = self.count_encoder_list.keys()
        index_features = self.index_encoder_list.keys()

        test_df['dow'] = test_df['f_1'] % 7
        test_df = transform_newvalue(fdflag_features, test_df, self.new_value_encoder_list)
        test_df = transform_count_encoded_feature(count_features, test_df, self.count_encoder_list)
        test_df = transform_indexing_feature(index_features, test_df, self.index_encoder_list)

        if enable_numpy:
            return test_df[self.output_feature_names].values
        else:
            return test_df[self.output_feature_names]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="/dataset", type=str)
    parser.add_argument("--feat_dir", dest="feat_dir", default="/feature_store", type=str)
    parser.add_argument("--save_dir", dest="save_dir", default="/output", type=str)
    parser.add_argument("--train", dest="is_train", action="store_true", default=False)
    parser.add_argument("--predict", dest="is_predict", action="store_true", default=False)
    args = parser.parse_args()
    
    dataprep = DataPrep(feat_dir = args.feat_dir)
    
    if args.is_train:
        dataprep.fit_transform(data_path = args.data_dir, data_save_dir=args.save_dir)

    elif args.is_predict:
        test_df = load_csv_to_pandasdf(args.data_dir).reset_index(drop=True)
        dataprep.prepare_features()
        processed_test_df = dataprep.transform(test_df)
        processed_test_df.to_parquet(os.path.join(args.save_dir,  "test_processed.parquet"))

    else:  
        raise ValueError(f'Please either use --train or --predict flag')
