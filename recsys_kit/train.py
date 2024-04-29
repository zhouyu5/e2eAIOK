import os
import sys
import datasets
from datasets import load_from_disk
from recsys_kit import *
from recsys_kit.models import *
from transformers import Trainer


def main():
    other_args, training_args = parse_args()
    training_args = init_training_args(training_args, other_args)

    dataset = load_from_disk(other_args.preprocess_dataset_path)
    train_ds = dataset[datasets.Split.TRAIN]
    valid_ds = dataset[datasets.Split.VALIDATION]
    if other_args.dev_mode:
        train_ds = train_ds.select(range(1000))
        valid_ds = valid_ds.select(range(1000))

    model_config = AutoConfig.from_pretrained(other_args.model_config_name_or_path)
    rec_model = AutoModel.from_config(model_config)

    if other_args.use_hpu:
        from optimum.habana import GaudiTrainer
        choose_trainer = GaudiTrainer
    else:
        choose_trainer = Trainer
    trainer = choose_trainer(
        model=rec_model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=valid_ds if training_args.do_eval else None,
        compute_metrics=compute_metrics(other_args.metrics),
    )
    
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_ds)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(valid_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
