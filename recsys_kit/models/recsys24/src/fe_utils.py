import numpy as np
import os
from tqdm import tqdm
from category_encoders.count import CountEncoder as SKLCountEncoder
from category_encoders import *
import pickle

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

class Indexer:
    def __init__(self, feature_name, partition_key, timestamp = 'f_1', counter = None):
        self.partition_key = partition_key
        self.timestamp = timestamp
        self.feature_name = feature_name
        if isinstance(counter, type(None)):
            self.counter = {}
        else:
            self.counter = counter
        
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls(data['feature_name'], data['partition_key'], data['timestamp'], data['indexer'])
        
    def save(self, filename):
        data = {}
        data['partition_key'] = self.partition_key
        data['timestamp'] = self.timestamp
        data['feature_name'] = self.feature_name
        data['indexer'] = self.counter
        
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        

    def fit_transform(self, df):
        new_features = []
        partition_key = self.partition_key
        feature_name = self.feature_name
        timestamp = self.timestamp
        day_range = df[timestamp].unique()
        day_range.sort()
        time_col = df[timestamp].to_list()
        feature_col = df[feature_name].to_list()
        partition_col = df[partition_key].to_list()        
        for time_value, feature_value, partition_value in zip(time_col, feature_col, partition_col):
            #print(time_value, feature_value, partition_value)
            if feature_value not in self.counter:
                self.counter[feature_value] = {}
            if partition_value not in self.counter[feature_value]:
                self.counter[feature_value][partition_value] = dict((k, 0) for k in day_range)       
            self.counter[feature_value][partition_value][time_value] += 1
            new_features.append(self.counter[feature_value][partition_value][time_value])
        return pd.Series(new_features, index = df.index)
    
    def transform(self, df):
        new_features = []
        partition_key = self.partition_key
        feature_name = self.feature_name
        timestamp = self.timestamp
        time_col = df[timestamp].to_list()
        feature_col = df[feature_name].to_list()
        partition_col = df[partition_key].to_list()        
        for time_value, feature_value, partition_value in zip(time_col, feature_col, partition_col):
            #print(time_value, feature_value, partition_value)
            if feature_value not in self.counter:
                self.counter[feature_value] = {}
            if partition_value not in self.counter[feature_value]:
                self.counter[feature_value][partition_value] = {}
            if time_value not in self.counter[feature_value][partition_value]:
                self.counter[feature_value][partition_value][time_value] = 0
            self.counter[feature_value][partition_value][time_value] += 1
            new_features.append(self.counter[feature_value][partition_value][time_value])
        return pd.Series(new_features, index = df.index)
    
class NewValueEncoder:
    def __init__(self, feature_name, encoder = None):
        self.feature_name = feature_name
        if isinstance(encoder, type(None)):
            self.encoder = None
        else:
            self.encoder = encoder
            
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls(data['feature_name'], data['encoder'])
        
    def save(self, filename):
        data = {}
        data['feature_name'] = self.feature_name
        data['encoder'] = self.encoder
        
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def fit_transform(self, df):
        feature_name = self.feature_name
        encoder = pd.DataFrame({f'{feature_name}_first_day':df.groupby([feature_name])['f_1'].min()})
        self.encoder = encoder
        df = df.merge(encoder, on = feature_name, how = 'left')
        df[f'{feature_name}_fdflag'] = (df['f_1'] == df[f'{feature_name}_first_day'])
        return df
        
    def transform(self, df):
        encoder_df = self.encoder
        feature_name = self.feature_name
        
        existing_values = encoder_df.index.to_list()
        df[f'{feature_name}_fdflag'] = ~df[feature_name].isin(existing_values)
        return df

class CountEncoder:
    def __init__(self, feature_name = None, handle_unknown = None, encoder = None):
        self.feature_name = feature_name
        if isinstance(encoder, type(None)):
            self.encoder = SKLCountEncoder(cols=[feature_name], handle_unknown=handle_unknown)
        else:
            self.encoder = encoder
        
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls(encoder = data['encoder'])
        
    def save(self, filename):
        data = {}
        data['encoder'] = self.encoder
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        

    def fit_transform(self, df):
        return self.encoder.fit_transform(df)
    
    def transform(self, df):
        return self.encoder.transform(df)

def fit_count_encoded_feature(fg_list_1, df):
    encoder_list = {}
    for feature_name in tqdm(fg_list_1, desc='get_count_encoded_feature'):
        feature_name_CE = f"{feature_name}_CE"            
        encoder = CountEncoder(feature_name, "return_nan")
        df[feature_name_CE] = encoder.fit_transform(df[feature_name])        
        encoder_list[feature_name] = encoder
    return df, encoder_list

def fit_indexing_feature(fg_list_1, partition_key, df):
    encoder_list = {}
    for feature_name in tqdm(fg_list_1, desc='get_indexing_feature'):
        feature_name_index = f"{feature_name}_idx"            
        encoder = Indexer(feature_name, partition_key)
        df[feature_name_index] = encoder.fit_transform(df)        
        encoder_list[feature_name] = encoder
    return df, encoder_list

def fit_newvalue(categorical_list, df):
    encoder_list = {}
    for feature_name in tqdm(categorical_list, desc='get_newvalue_flag_feature'):
        encoder = NewValueEncoder(feature_name)
        df = encoder.fit_transform(df)
        encoder_list[feature_name] = encoder
    return df, encoder_list
    
def transform_count_encoded_feature(fg_list_1, df, encoder_list):
    for feature_name in tqdm(fg_list_1, desc='get_count_encoded_feature'):
        feature_name_CE = f"{feature_name}_CE"        
        encoder = encoder_list[feature_name]
        df[feature_name_CE] = encoder.transform(df[feature_name])        
        df[feature_name_CE] = df[feature_name_CE].fillna(1)
    return df

def transform_indexing_feature(fg_list_1, df, encoder_list):
    for feature_name in tqdm(fg_list_1, desc='get_indexing_feature'):
        feature_name_index = f"{feature_name}_idx"            
        encoder = encoder_list[feature_name]
        df[feature_name_index] = encoder.transform(df)
    return df

def transform_newvalue(categorical_list, df, encoder_list):
    for feature_name in tqdm(categorical_list, desc='get_newvalue_flag_feature'):        
        encoder = encoder_list[feature_name]
        df = encoder.transform(df)
    return df
