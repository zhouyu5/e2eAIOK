from transformers import PretrainedConfig
from typing import List, Optional
import polars as pl
import polars.selectors as cs
from datasets import Dataset
import numpy as np

from recsys_kit.feature import CAT_INDEX_SHIFT, read_json_key


class TF4CTRConfig(PretrainedConfig):
    model_type = "tab_tf_ctr"

    def __init__(
        self,
        feature_config_name_or_path: Optional[str] = None,
        hidden_dim: int = 32,
        num_classes: int = 2,
        n_layers: int = 6,
        n_heads: int = 8,
        attention_dropout: float = 0.1,
        dropout: float = 0.1,
        mlp_hidden_mults: List[int] = [4, 2],
        activation: str = "relu",
        **kwargs,
    ):
        self.feature_config_name_or_path = feature_config_name_or_path
        self.categories = read_json_key(self.feature_config_name_or_path, 'cat_cardinality')
        self.num_continuous = read_json_key(self.feature_config_name_or_path, 'dense_feat_num')
        self.cat_columns = read_json_key(self.feature_config_name_or_path, 'cat_columns')
        self.dense_columns = read_json_key(self.feature_config_name_or_path, 'dense_columns')
        self.label_column = read_json_key(self.feature_config_name_or_path, 'label_column')
        self.hidden_dim = hidden_dim
        self.dim_out = num_classes - 1
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.mlp_hidden_mults = mlp_hidden_mults
        self.activation = activation
        
        super().__init__(**kwargs)


    def format_model_input_all(self, dataset):
        for partition_key in dataset.keys():
            print(f"begin to postprocess partition {partition_key}")
            df = (
                dataset[partition_key].to_polars().lazy()
                .with_columns((pl.col(self.cat_columns)+CAT_INDEX_SHIFT))
                .with_columns(pl.concat_list(self.cat_columns).alias('cat_tensor'))
                .with_columns(pl.concat_list(self.dense_columns).alias('dense_tensor'))
                .with_columns(pl.concat_list([self.label_column]).alias('labels'))
                .drop(~cs.contains(("cat_tensor", "dense_tensor", "labels")))
                .collect()
                .to_pandas()
            )
            dataset[partition_key] = Dataset.from_pandas(df)
            print(f"finish postprocessing partition {partition_key}")
        return dataset


    def format_model_input_batch(self, examples, num_cont, num_cat):
        """
        usage example:
        cat_feat_num = len(other_args.cat_columns.split(','))
        dense_feat_num = len(other_args.dense_columns.split(','))
        
        print(dataset)
        dataset_random_key = list(dataset.keys())[0]
        dataset = dataset.map(
            lambda x: feature_format(x, dense_feat_num, cat_feat_num), 
            batched=True, 
            remove_columns=dataset[dataset_random_key].column_names, 
            num_proc=8
        )
        """
        rt = []
        for key in examples:
            rt.append(examples[key])
        rt = np.array(rt)
        rt = np.transpose(rt)

        dense_tensors = rt[:, :num_cont]
        cat_tensors = rt[:, num_cont:(num_cont+num_cat)]
        labels = rt[:, -1]
        
        return {
            'cat_tensor': cat_tensors.astype(int),
            'dense_tensor': dense_tensors,
            'labels': np.expand_dims(labels.astype(int), axis=1)
        }