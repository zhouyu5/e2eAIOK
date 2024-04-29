import os
import pickle
import argparse

from pathlib import Path
import polars as pl

from ebrec.utils._descriptive_analysis import (
    min_max_impression_time_behaviors, 
    min_max_impression_time_history
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL, 
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL, 
    DEFAULT_USER_COL,
    DEFAULT_ARTICLE_ID_COL
)

from transformers import AutoTokenizer, AutoModel
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import convert_text2encoding_with_transformers, create_article_id_to_value_mapping

from recsys_kit.utils import Timer

def ebnerd_from_path(path:Path, history_size:int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function 
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes, df2=df_history.collect(), on=DEFAULT_USER_COL, how="left"
        )
    )
    return df_behaviors

class ArticleClickDataPrep:
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        
    def prepare_features(self):
        pass
    
    def fit_transform(self, data_path, data_save_dir, LOCAL_MODEL_CACHE):
        # 1. prepare output dirs
        data_output_dir = os.path.join(data_save_dir, "processed")
        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir, exist_ok=True)
        
        feat_save_dir = self.feat_dir
        if not os.path.exists(feat_save_dir):
            os.makedirs(feat_save_dir, exist_ok=True)

        path = Path(data_path)

        # 2. actual data process
        COLUMNS = [DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL]
        HISTORY_SIZE = 30
        N_SAMPLES = 100
        TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
        MAX_TITLE_LENGTH = 30
        TRANSFORMER_MODEL_NAME = "bert-base-multilingual-cased"
        
        # 2.1 Train 
        with Timer("Process Train data"):
            df_train = (
                ebnerd_from_path(path.joinpath("train"), history_size=HISTORY_SIZE)
                .select(COLUMNS)
                .pipe(sampling_strategy_wu2019,npratio=4,shuffle=True,with_replacement=True, seed=123)
                .pipe(create_binary_labels_column)
                .sample(n=N_SAMPLES)
            )
        with Timer("Save Train to Disk"):
            df_train.write_parquet(os.path.join(data_output_dir, "train_processed.parquet"))

        # 2.2 Valid
        with Timer("Process Validate data"):
            df_validation = (
                ebnerd_from_path(path.joinpath("validation"), history_size=HISTORY_SIZE)
                .select(COLUMNS)
                .pipe(create_binary_labels_column)
                .sample(n=N_SAMPLES)
            )
        with Timer("Save Valid to Disk"):
            df_validation.write_parquet(os.path.join(data_output_dir, "valid_processed.parquet"))
        
        # 2.3 Articles
        with Timer("Process Articles"):
            df_articles = pl.read_parquet(path.joinpath("articles.parquet"))
            transformer_tokenizer = AutoTokenizer.from_pretrained(os.path.join(LOCAL_MODEL_CACHE, TRANSFORMER_MODEL_NAME))
            # LOAD HUGGINGFACE:
            df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
            df_articles, token_col_title = convert_text2encoding_with_transformers(
                df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
            )
            df_articles = df_articles.with_columns(pl.Series(TRANSFORMER_MODEL_NAME, df_articles[token_col_title]))
            df_articles = df_articles.drop(token_col_title)
            article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=TRANSFORMER_MODEL_NAME)
        with Timer("Save Articles mapping to Disk"):
            with open(os.path.join(feat_save_dir, "article_mapping.pkl"), 'wb') as file:
                pickle.dump(article_mapping, file)

        # self.metadata = {}
        # with open(os.path.join(feat_save_dir, "meta.pkl"), 'wb') as file:
        #     pickle.dump(self.metadata, file)

    def transform(self, data):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="/dataset", type=str)
    parser.add_argument("--feat_dir", dest="feat_dir", default="/feature_store", type=str)
    parser.add_argument("--save_dir", dest="save_dir", default="/output", type=str)
    parser.add_argument("--local_model_dir", dest="local_model_dir", default="models/", type=str)
    parser.add_argument("--train", dest="is_train", action="store_true", default=False)
    parser.add_argument("--predict", dest="is_predict", action="store_true", default=False)
    args = parser.parse_args()
    
    dataprep = ArticleClickDataPrep(feat_dir = args.feat_dir)
    
    if args.is_train:
        dataprep.fit_transform(data_path = args.data_dir, data_save_dir=args.save_dir, LOCAL_MODEL_CACHE=args.local_model_dir)

    elif args.is_predict:
        pass

    else:  
        raise ValueError(f'Please either use --train or --predict flag')





