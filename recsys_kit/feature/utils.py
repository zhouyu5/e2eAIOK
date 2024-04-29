import json
from sklearn.compose import make_column_selector

from .constants import DENSE_FEAT_PREFIX, CAT_FEAT_PREFIX, CAT_INDEX_SHIFT, SAMPLE_NUM_TO_DEBUG


def sample_dataset(dataset):
    for key in dataset.keys():
        sample_num = min(SAMPLE_NUM_TO_DEBUG, len(dataset[key]))
        dataset[key] = dataset[key].select(range(sample_num))
    return dataset


def get_cardinality_from_data_pipeline(process_pipe):
    encoder = process_pipe.transformers_[1][1].named_steps['categorify']
    return tuple(len(item)+CAT_INDEX_SHIFT for item in encoder.categories_)


def get_columns_from_dataset(dataset, partition_key):
    dataset.set_format("pandas")
    df = dataset[partition_key][:1]
    all_dense_columns = make_column_selector(pattern=DENSE_FEAT_PREFIX)(df)
    all_cat_columns = make_column_selector(pattern=CAT_FEAT_PREFIX)(df)
    dataset.reset_format()
    return all_cat_columns, all_dense_columns


def append_key_value_to_json(file_path, key, value):
    with open(file_path, 'r+') as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()     # remove remaining part

def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_json_key(file_path, key):
    if not file_path:
        return None
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data.get(key, None)
    

def rename_columns_with_pattern(df, pattern="remainder"):
    reminder_columns = make_column_selector(pattern=pattern)(df)
    reminder_columns_map = {
        item: item.replace(f"{pattern}__", "")
        for item in reminder_columns
    }
    df = df.rename(columns=reminder_columns_map)
    return df