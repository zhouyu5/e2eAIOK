import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser


@dataclass
class RecTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="checkpoint/tmp",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


@dataclass
class OtherArguments:
    # hardware related config
    use_hpu: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use habana gaudi2 for training"
                "should only be set to `True` if you have gaudi2."
            )
        },
    )
    # config file
    model_config_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path or name of the model config file to use."}
    )
    train_config_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path or name of the training config file to use."}
    )
    feature_config_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path or name of the feature config file to use."}
    )
    gaudi_config_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The name or path of the Gaudi config file to use."}
    )
    # data related file
    dataset_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path or name of the hugging face dataset."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The config name of the hugging face dataset."}
    )
    data_pipeline_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path of the data preprocess pipeline."}
    )
    download_data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The local path of the downloaded data."}
    )
    preprocess_dataset_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The local path of the preprocess dataset."}
    )
    # feature related config
    fit_transform_output_type: str = field(
        default="polars", 
        metadata={"help": "The output type of the preprocessor's transform output, options include pandas, polars."}
    )
    dense_columns: Optional[str] = field(
        default=None, 
        metadata={"help": "The column names of the dense features, comma seperated."}
    )
    cat_columns: Optional[str] = field(
        default=None, 
        metadata={"help": "The column names of the categorical features, comma seperated."}
    )
    label_column: Optional[str] = field(
        default=None, 
        metadata={"help": "The column name of the label."}
    )
    dense_processor: Optional[str] = field(
        default=None, 
        metadata={"help": "The processor names for the dense features, comma seperated."}
    )
    cat_processor: Optional[str] = field(
        default=None, 
        metadata={"help": "The processor names for the cat features, comma seperated."}
    )
    # metric related config
    metrics: Optional[str] = field(
        default=None, 
        metadata={"help": "The list of metrics to use, comma seperated."}
    )
    # develop related config
    dev_mode: bool = field(
        default=False,
        metadata={
            "help": ("Whether or not under development mode, will use smaller dataset for faster debugging.")
        },
    )


def parse_args():
    parser = HfArgumentParser((OtherArguments, RecTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        other_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        other_args, training_args = parser.parse_args_into_dataclasses()
    return other_args, training_args


def init_training_args(training_args, other_args):
    training_args_dict = training_args.to_dict()
    if not other_args.use_hpu:
        training_args = TrainingArguments(**training_args_dict)
    else:
        from optimum.habana import GaudiTrainingArguments
        training_args_dict.update({
            "use_habana": True,
            "use_lazy_mode": True,
            "gaudi_config_name": other_args.gaudi_config_name_or_path,
        })
        training_args = GaudiTrainingArguments(**training_args_dict)
    return training_args