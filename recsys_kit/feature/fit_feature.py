import datasets
from datasets import load_from_disk, load_dataset

from recsys_kit import *
from .pipe import get_preprocess_pipeline
from .utils import get_cardinality_from_data_pipeline, write_json, sample_dataset


def init_feat_config(process_pipe, other_args):
    data = {
        'cat_cardinality': get_cardinality_from_data_pipeline(process_pipe),
        'dense_feat_num': len(process_pipe.transformers_[0][2]),
        'cat_columns' : other_args.cat_columns.split(','),
        'dense_columns' : other_args.dense_columns.split(','),
        'label_column': other_args.label_column,
    }
    write_json(other_args.feature_config_name_or_path, data)
    return other_args


def main():
    other_args, _ = parse_args()
    
    dataset = load_dataset(
        other_args.dataset_name_or_path, 
        other_args.dataset_config_name,
        cache_dir=other_args.download_data_path, 
        trust_remote_code=True,
        num_proc=8
    )

    print('raw dataset:')
    print(dataset)

    if other_args.dev_mode:
        dataset = sample_dataset(dataset)
    
    process_pipe = get_preprocess_pipeline(dataset, datasets.Split.TRAIN, other_args)

    init_feat_config(process_pipe, other_args)


if __name__ == "__main__":
    main()
