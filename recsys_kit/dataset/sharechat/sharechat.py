# coding=utf-8
"""The dataset corresponds to roughly 10M random users who visited the ShareChat + Moj app over three months. 
We have sampled each user's activity to generate 10 impressions corresponding to each user. 
Our target variable is whether there was an install for an app by the user or not.
"""


import csv
import json
import os
import glob
import polars as pl

import datasets


_CITATION = """\
@incollection{agrawal2023recsys,
  title={RecSys Challenge 2023 Dataset: Ads Recommendations in Online Advertising},
  author={Agrawal, Rahul and Brahme, Sarang and Maitra, Sourav and Srivastava, Abhishek and Irissappane, Athirai and Liu, Yong and Kalloori, Saikishore},
  booktitle={Proceedings of the Recommender Systems Challenge 2023},
  pages={1--3},
  year={2023}
}
"""

_DESCRIPTION = """\
The dataset corresponds to roughly 10M random users who visited the ShareChat + Moj app over three months. 
We have sampled each user's activity to generate 10 impressions corresponding to each user. 
Our target variable is whether there was an install for an app by the user or not.
"""

_HOMEPAGE = "https://www.recsyschallenge.com/2023/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""


_URLS = {
    "first_domain": "https://cdn.sharechat.com/2a161f8e_1679936280892_sc.zip",
}


class Sharechat(datasets.ArrowBasedBuilder):
    """The dataset for RecSys Challenge 2024."""

    VERSION = datasets.Version("1.0.0")

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
    ]

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            # data format: f0,f1,...,f79, is_clicked, is_installed
            # all_cat_columns = time_columns + cat_columns + binary_columns
            id_columns = [('f_0', datasets.Value("int64"))]
            time_columns = [('f_1', datasets.Value("int8"))]
            cat_columns = [(f'f_{i}', datasets.Value("int32")) for i in range(2, 33)]
            binary_columns = [(f'f_{i}', datasets.Value("int8")) for i in range(33, 42)]
            dense_columns = [(f'f_{i}', datasets.Value("float")) for i in range(42, 80)]
            other_columns = [('is_clicked', datasets.Value("int8"))]
            label_columns = [('is_installed', datasets.Value("int8"))]
            all_columns = id_columns + time_columns + cat_columns + binary_columns + dense_columns + other_columns + label_columns
            
            features = datasets.Features(dict(all_columns))
        else:  # This is an example to show how to have different features for other domains
            raise NotImplementedError("This configuration is not implemented yet")
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        id_columns = [('f_0', pl.Int64)]
        time_columns = [('f_1', pl.Int8)]
        cat_columns = [(f'f_{i}', pl.Int32) for i in range(2, 33)]
        binary_columns = [(f'f_{i}', pl.Int8) for i in range(33, 42)]
        dense_columns = [(f'f_{i}', pl.Float32) for i in range(42, 80)]
        other_columns = [('is_clicked', pl.Int8)]
        label_columns = [('is_installed', pl.Int8)]
        all_columns = id_columns + time_columns + cat_columns + binary_columns + dense_columns + other_columns + label_columns
        self.dtypes_dict = dict(all_columns)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": sorted(glob.glob(os.path.join(data_dir, "sharechat_recsys2023_data", "train", "*.csv"))),
                    "date_start": 45,
                    "date_end": 64,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": sorted(glob.glob(os.path.join(data_dir, "sharechat_recsys2023_data", "train", "*.csv"))),
                    "date_start": 65,
                    "date_end": 65,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": sorted(glob.glob(os.path.join(data_dir, "sharechat_recsys2023_data", "train", "*.csv"))),
                    "date_start": 66,
                    "date_end": 66,
                },
            ),
        ]

    def _generate_tables(self, filepaths, date_start, date_end):
        # This method handles input defined in _split_generators to yield (key, table) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for id, filepath in enumerate(filepaths):
            if self.config.name == "first_domain":
                pa_table = (
                    pl.scan_csv(filepath, separator='\t', dtypes=self.dtypes_dict)
                    .filter(pl.col("f_1").is_between(date_start, date_end))
                    .sort("f_1", descending=False)
                    .with_columns(pl.col("f_1").mod(7).alias("f_1"))
                    .collect().to_arrow()
                )
                yield id, pa_table
            else:
                raise NotImplementedError



# test the script
# datasets-cli test sharechat.py --save_info --all_configs
# share the dataset
# huggingface-cli login
# huggingface-cli repo create sharechat-dataset --type dataset
# huggingface-cli upload sharechat-dataset ./sharechat/ . --repo-type dataset
