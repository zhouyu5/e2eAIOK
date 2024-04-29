import os
import pickle

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn import set_config

from .cat import CAT_ENCODER_MAP
from .dense import DENSE_ENCODER_MAP
from .constants import *


def get_encoder(encoder_key_list):
    ENCODER_MAP = {**CAT_ENCODER_MAP, **DENSE_ENCODER_MAP}
    steps = [
        (key, ENCODER_MAP[key])
        for key in encoder_key_list
    ]
    enc = Pipeline(steps=steps, verbose=True)
    return enc


def get_preprocess_pipeline(dataset, partition_key, other_args):
    set_config(transform_output=other_args.fit_transform_output_type)

    dataset.set_format(other_args.fit_transform_output_type)
    df = dataset[partition_key][:]

    dense_columns = other_args.dense_columns.split(',')
    cat_columns = other_args.cat_columns.split(',')
    save_path = other_args.data_pipeline_path

    cat_enc = get_encoder(other_args.cat_processor.split(','))
    dense_enc = get_encoder(other_args.dense_processor.split(','))
    
    feature_process = ColumnTransformer(
        transformers=[
            (DENSE_FEAT_PREFIX, dense_enc, dense_columns),
            (CAT_FEAT_PREFIX, cat_enc, cat_columns),
        ],
        remainder='passthrough',
        n_jobs=8,
        verbose=True,
        verbose_feature_names_out=False,
    )

    print('begin to fit the feature dataframe')
    feature_process.fit(df)

    print('begin to save the process pipeline')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(feature_process, open(save_path, 'wb'))
    print('the process pipeline is successfully saved')
    
    return feature_process