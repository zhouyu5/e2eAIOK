import os
import sys
import pickle
from datasets import Dataset, load_dataset, disable_caching
from sklearn import set_config
from recsys_kit.models import *

from recsys_kit import *
from .utils import sample_dataset

def preprocess_df(df, process_pipe=None, save_path=None):
    if save_path:
        process_pipe = pickle.load(open(save_path, 'rb'))
    # print(process_pipe.get_feature_names_out())
    print(df.head())
    df = process_pipe.transform(df)
    print(df.head())
    return df


def preprocess_dataset(dataset, other_args):
    pipeline_save_path = other_args.data_pipeline_path
    transform_output_type = other_args.fit_transform_output_type
    set_config(transform_output=transform_output_type)
    for partition_key in dataset.keys():
        print(f"begin to preprocess partition {partition_key}")
        dataset.set_format(transform_output_type)
        df = dataset[partition_key][:]
        df = preprocess_df(df, save_path=pipeline_save_path)
        if transform_output_type == "pandas":
            dataset[partition_key] = Dataset.from_pandas(df)
        elif transform_output_type == "polars":
            dataset[partition_key] = Dataset.from_polars(df)
        else:
            raise NotImplementedError(f"transform_output_type {transform_output_type} is not supported")
        dataset.reset_format()
        print(f"finish preprocessing partition {partition_key}")
    return dataset


def main():
    other_args, _ = parse_args()
    if other_args.dev_mode:
        disable_caching()

    dataset = load_dataset(
        other_args.dataset_name_or_path, 
        other_args.dataset_config_name,
        cache_dir=other_args.download_data_path, 
        trust_remote_code=True,
        num_proc=8,
    )
    if other_args.dev_mode:
        dataset = sample_dataset(dataset)

    dataset = preprocess_dataset(dataset, other_args)

    model_config = AutoConfig.from_pretrained(other_args.model_config_name_or_path)

    dataset = model_config.format_model_input_all(dataset)

    print('preprocessed dataset:')
    print(dataset)
    print(dataset[list(dataset.keys())[0]][0])
    dataset.save_to_disk(other_args.preprocess_dataset_path)


if __name__ == "__main__":
    main()