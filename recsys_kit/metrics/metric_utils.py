from transformers import EvalPrediction
import evaluate
import os


def compute_metrics(metrics):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    metric_list = [f'{dir_path}/{metric}' for metric in metrics.split(",")] if metrics else None
    if metric_list:
        multi_metrics = evaluate.combine(metric_list)
        return lambda p: multi_metrics.compute(prediction_scores=p.predictions, references=p.label_ids)
    else:
        return None
