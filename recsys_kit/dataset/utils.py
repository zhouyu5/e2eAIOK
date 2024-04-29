from datasets import load_dataset, Dataset, disable_caching


def load_csv_dataset(data_dir, sep='\t', num_proc=8):
    dataset = load_dataset(
        "csv", sep=sep, 
        data_dir=data_dir, 
        num_proc=num_proc
    )
    return dataset
