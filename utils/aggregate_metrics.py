import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from mlflow.tracking import MlflowClient
from prettytable import PrettyTable
from tqdm import tqdm

"""
Steps to successfully use this script:
1. Correct the artifact uri running the script yamlAdapt.py
2. Open the mlflow ui and delete unwanted models
3. Run this script to get a txt file with aggregated metrics
"""

metrics = [
    'inference_time',
    'kappa_score',
    'precision',
    'recall',
    'f1score',
    'oacc',
    'rocauc',
    'acc_perclass',
    'precision_perclass',
    'recall_perclass',
    'f1score_perclass',
    'roc_class']


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Aggregate cross-validation test metrics',
        add_help=False)

    parser.add_argument(
        '--trackingUri',
        default="file://" + os.getcwd() + "/mlruns_final_complete",
        type=str,
        help='Tracking URI')
    parser.add_argument(
        '--fileToSave',
        default="final_metrics.txt",
        type=str,
        help='File to save the metrics')
    parser.add_argument(
        '--nfolds',
        default=5,
        type=int,
        help='Number of fold used for cross-validation')
    parser.set_defaults()

    return parser


def find_json_files_in_directory(directory_uri):
    json_files = []
    directory_path = Path(urlparse(directory_uri).path)

    for json_file in directory_path.rglob("*.json"):
        json_files.append(str(json_file))

    return json_files


def load_metrics_from_run(directory_uri):
    # metrics are store in a json file during test phase
    json_files = find_json_files_in_directory(directory_uri)

    metrics = {
        "inference_time": [],
        "kappa_score": [],
        "precision": [],
        "recall": [],
        "f1score": [],
        "oacc": [],
        "rocauc": [],
        "acc_perclass": [],
        "precision_perclass": [],
        "recall_perclass": [],
        "f1score_perclass": [],
        "roc_class": []
    }

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)

            # Global metrics
            metrics["inference_time"].append(data.get("inference_time", 0))
            metrics["kappa_score"].append(data.get("kappa_score", 0))
            metrics["precision"].append(data.get("precision", 0))
            metrics["recall"].append(data.get("recall", 0))
            metrics["f1score"].append(data.get("f1score", 0))
            metrics["oacc"].append(data.get("oacc", 0))
            metrics["rocauc"].append(data.get("rocauc", 0))

            # Per class metrics
            metrics["acc_perclass"].append(data.get("per_class_accuracy", 0))
            metrics["precision_perclass"].append(
                data.get("precision_class", 0))
            metrics["recall_perclass"].append(data.get("recall_class", 0))
            metrics["f1score_perclass"].append(data.get("fscore_class", 0))
            metrics["roc_class"].append(data.get("roc_class", 0))

    return metrics


def global_metrics(model_metrics, nfolds):
    aggregated_results = {}

    for db_name, jobs in model_metrics.items():
        if db_name not in aggregated_results:
            aggregated_results[db_name] = {}

        for job_name, seed_metrics in jobs.items():
            all_metrics = {}

            for seed in range(nfolds):
                if str(seed) in seed_metrics:
                    for metric in seed_metrics[str(seed)]:
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].extend(
                            seed_metrics[str(seed)].get(metric, []))

            aggregated_results[db_name][job_name] = {
                "num_params": model_metrics[db_name][job_name]["num_params"]}

            for metric_name, values in all_metrics.items():
                if not values:
                    continue

                aggregated_results[db_name][job_name].setdefault(
                    metric_name, {})
                
                multiplier = 1 if metric_name == "inference_time" else 100

                if isinstance(values, list):
                    values = np.array(values)
                    filtered_values = np.where(values == -1, np.nan, values)
                    aggregated_results[db_name][job_name][metric_name] = {
                        "mean": np.nanmean(filtered_values, axis=0) * multiplier,
                        "stddev": np.nanstd(filtered_values, axis=0) * multiplier
                    }
                else:
                    aggregated_results[db_name][job_name][metric_name] = {  # single value, not a list. It should never happen unless the test set has a single image
                        "mean": values,
                        "stddev": 0
                    }

    return aggregated_results


def fold_metrics(model_metrics, nfolds):
    fm = {}

    for db_name, jobs in model_metrics.items():
        if db_name not in fm:
            fm[db_name] = {}

        for job_name, seed_metrics in jobs.items():
            if job_name not in fm[db_name]:
                fm[db_name][job_name] = {}

            for seed in range(nfolds):
                if str(seed) in seed_metrics:
                    for metric in seed_metrics[str(seed)]:
                        metric_values = seed_metrics[str(seed)].get(metric, [])

                        fm.setdefault(
                            db_name,
                            {}).setdefault(
                            job_name,
                            {}).setdefault(
                            str(seed),
                            {}).setdefault(
                            metric,
                            {})

                        if not metric_values:
                            continue

                        if isinstance(metric_values, list):
                            metric_values = np.array(metric_values)
                            filtered_values = np.where(
                                metric_values == -1, np.nan, metric_values)
                            fm[db_name][job_name][str(seed)][metric] = {
                                "mean": np.nanmean(filtered_values, axis=0) * 100,
                                "stddev": np.nanstd(filtered_values, axis=0) * 100
                            }
                        else:
                            fm[db_name][job_name][str(seed)][metric] = {
                                # single value, not a list. It should never
                                # happen unless the test set has a single image
                                "mean": metric_values,
                                "stddev": 0
                            }
    return fm


def main(args):
    os.environ['MLFLOW_TRACKING_URI'] = args.trackingUri
    nfolds = args.nfolds

    client = MlflowClient()
    experiments = client.search_experiments()

    table_global = PrettyTable()
    table_global.field_names = [
        "Dataset",
        "Model",
        "N. Params",
        "Inference time (s)",
        'Kappa score',
        'Precision',
        'Recall',
        'F1',
        'OACC',
        'ROCAUC',
        "",
        "ACC Healthy",
        "ACC Tumor",
        "ACC Blood",
        "ACC Dura",
        "Precision Healthy",
        "Precision Tumor",
        "Precision Blood",
        "Precision Dura",
        "Recall Healthy",
        "Recall Tumor",
        "Recall Blood",
        "Recall Dura",
        "F1 Healthy",
        "F1 Tumor",
        "F1 Blood",
        "F1 Dura",
        "AUC Healthy",
        "AUC Tumor",
        "AUC Blood",
        "AUC Dura"]

    fold_tables = [PrettyTable() for _ in range(nfolds)]
    for f, table_fold in enumerate(fold_tables):
        table_fold.field_names = [
            "Dataset",
            "Model",
            f'F{f} Kappa score',
            f'F{f} Precision',
            f'F{f} Recall',
            f'F{f} F1',
            f'F{f} OACC',
            f'F{f} ROCAUC',
            "",
            f'F{f} ACC Healthy',
            f'F{f} ACC Tumor',
            f'F{f} ACC Blood',
            f'F{f} ACC Dura',
            f'F{f} Precision Healthy',
            f'F{f} Precision Tumor',
            f'F{f} Precision Blood',
            f'F{f} Precision Dura',
            f'F{f} Recall Healthy',
            f'F{f} Recall Tumor',
            f'F{f} Recall Blood',
            f'F{f} Recall Dura',
            f'F{f} F1 Healthy',
            f'F{f} F1 Tumor',
            f'F{f} F1 Blood',
            f'F{f} F1 Dura',
            f'F{f} AUC Healthy',
            f'F{f} AUC Tumor',
            f'F{f} AUC Blood',
            f'F{f} AUC Dura']

    models_metrics = {}
    for experiment in tqdm(experiments, desc="Working on experiments data"):
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        for run in runs:
            db_name = run.data.params.get("db_name", None)
            job_name = run.data.params.get("job_name", None).strip()
            seed = run.data.params.get("seed", None)
            n_params = run.data.params.get("n_parameters", 0)

            artifact_uri = run.info.artifact_uri
            metrics = load_metrics_from_run(artifact_uri)

            if db_name is not None:
                if job_name is not None:
                    if seed is not None:
                        if db_name not in models_metrics:
                            models_metrics[db_name] = {}
                        if job_name not in models_metrics[db_name]:
                            models_metrics[db_name][job_name] = {
                                "num_params": n_params}
                        models_metrics[db_name][job_name][seed] = metrics

    gm = global_metrics(models_metrics, nfolds)
    fm = fold_metrics(models_metrics, nfolds)

    for db_name, jobs in gm.items():
        for job_name, job_metrics in jobs.items():
            table_global.add_row([
                db_name, job_name, 
                job_metrics.get('num_params', 0),
                f"{job_metrics.get('inference_time', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('inference_time', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                f"{job_metrics.get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                f"{job_metrics.get('precision', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('precision', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                f"{job_metrics.get('recall', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('recall', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                f"{job_metrics.get('f1score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('f1score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                f"{job_metrics.get('oacc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('oacc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                f"{job_metrics.get('rocauc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {job_metrics.get('rocauc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                "",
                f"{job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                f"{job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                f"{job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                f"{job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {job_metrics.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                f"{job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                f"{job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                f"{job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                f"{job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {job_metrics.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                f"{job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                f"{job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                f"{job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                f"{job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {job_metrics.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                f"{job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                f"{job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                f"{job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                f"{job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {job_metrics.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                f"{job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                f"{job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                f"{job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                f"{job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {job_metrics.get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}"
            ])

    for db_name, jobs in fm.items():
        for job_name, folds in jobs.items():
            for fold in folds.keys():
                fold_tables[int(fold)].add_row([
                    db_name, job_name,
                    f"{folds[fold].get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {folds[fold].get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{folds[fold].get('precision', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {folds[fold].get('precision', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{folds[fold].get('recall', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {folds[fold].get('recall', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{folds[fold].get('f1score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {folds[fold].get('f1score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{folds[fold].get('oacc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {folds[fold].get('oacc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{folds[fold].get('rocauc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {folds[fold].get('rocauc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    "",
                    f"{folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {folds[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {folds[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {folds[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {folds[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {folds[fold].get('roc_class', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}"
                ])

    # sort according to db_name
    table_global.sortby = "Dataset"

    table_data_fold = []
    for f, table_fold in enumerate(fold_tables):
        table_fold.sortby = "Dataset"
        table_data_fold.append(table_fold)

    # PRINT TABLES
    file = open(args.fileToSave, "w")
    file.write(table_global.get_string())
    file.write("\n\n")
    file.write("Metrics per fold\n\n")
    for f, table_fold in enumerate(table_data_fold):
        file.write(f"Fold {f}\n")
        file.write(table_fold.get_string())
        file.write("\n\n")
    file.close()

    print("Metrics saved in: ", args.fileToSave)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
