import mlflow
from mlflow.tracking import MlflowClient
from prettytable import PrettyTable

import argparse
from urllib.parse import urlparse
from pathlib import Path

import numpy as np

import json
import os

from tqdm import tqdm

import pandas as pd

import math

metrics = ['kappa_score', 'precision', 'recall', 'f1score', 'oacc', 'rocauc', 'acc_perclass', 'precision_perclass', 'recall_perclass', 'f1score_perclass', 'auc_perclass']

def get_args_parser():
    parser = argparse.ArgumentParser('Aggregate cross-validation test metrics', add_help=False)

    parser.add_argument('--trackingUri', default="file:///home/domenico/mlruns_final", type=str, help='Tracking URI')
    parser.add_argument('--fileToSave', default="final_metrics.txt", type=str, help='File to save the metrics')
    parser.add_argument('--nfolds', default=3, type=int, help='Number of fold used for cross-validation')
    parser.set_defaults()

    return parser

def find_json_files_in_directory(directory_uri):
    json_files = []
    directory_path = Path(urlparse(directory_uri).path)
    
    for json_file in directory_path.rglob("*.json"):
        json_files.append(str(json_file))
    
    return json_files
    
def load_metrics_from_run(directory_uri):
    json_files = find_json_files_in_directory(directory_uri) #metrics are store in a json file during test phase
    
    metrics = {
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
        "auc_perclass": []
    }
    
    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            
            # Global metrics
            metrics["kappa_score"].append(data.get("kappa_score", 0))
            metrics["precision"].append(data.get("precision", 0))
            metrics["recall"].append(data.get("recall", 0))
            metrics["f1score"].append(data.get("f1score", 0))
            metrics["oacc"].append(data.get("oacc", 0))
            metrics["rocauc"].append(data.get("rocauc", 0))

            # Per class metrics
            metrics["acc_perclass"].append(data.get("per_class_accuracy", 0))
            metrics["precision_perclass"].append(data.get("precision_class",0))
            metrics["recall_perclass"].append(data.get("recall_class",0))
            metrics["f1score_perclass"].append(data.get("fscore_class",0))
            metrics["auc_perclass"].append(data.get("auc_c",0))
    
    return metrics
    
def global_metrics(model_metrics, nfolds):
    aggregated_results = {}

    for job_name, seed_metrics in model_metrics.items():
        all_metrics = {}
        for seed in range(nfolds):
            if str(seed) in seed_metrics:
                for metric in metrics:
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].extend(seed_metrics[str(seed)].get(metric, []))


        aggregated_results[job_name] = {}
        for metric_name, values in all_metrics.items():
            if values == []:
                continue

            aggregated_results.setdefault(job_name, {}).setdefault(metric_name, {})

            if isinstance(values[0], list):
                aggregated_results[job_name][metric_name] = {
                    "mean": np.mean(values, axis=1)*100,
                    "stddev": np.std(values, axis=1)*100
                }
            else:
                aggregated_results[job_name][metric_name] = {
                    "mean": np.mean(values) * 100,
                    "stddev": np.std(values) * 100
                }

    return aggregated_results

def fold_metrics(model_metrics, nfolds):
    fm = {}

    for job_name, seed_metrics in model_metrics.items():
        for seed in range(nfolds):
            if str(seed) in seed_metrics:
                for metric in metrics:
                    metric_values = seed_metrics[str(seed)].get(metric, [])

                    fm.setdefault(job_name, {}).setdefault(str(seed), {}).setdefault(metric, {})
                    
                    if metric_values == []:
                        continue

                    if isinstance(metric_values[0], list):
                        fm[job_name][str(seed)][metric] = {
                            "mean": np.mean(metric_values, axis=1) * 100,
                            "stddev": np.std(metric_values, axis=1) * 100
                        }
                    else:
                        fm[job_name][str(seed)][metric] = {
                            "mean": np.mean(metric_values) * 100,
                            "stddev": np.std(metric_values) * 100
                        }

    return fm

def main(args):
    os.environ['MLFLOW_TRACKING_URI'] = args.trackingUri
    nfolds = args.nfolds

    client = MlflowClient()
    experiments = client.search_experiments()

    table_global = PrettyTable()
    table_global.field_names = ["Dataset", "Model", 'Kappa score', 'Precision', 'Recall', 'F1', 'OACC', 'ROCAUC', "", "ACC Healthy", "ACC Tumor", "ACC Blood", "ACC Dura", "Precision Healthy", "Precision Tumor", "Precision Blood", "Precision Dura", "Recall Healthy", "Recall Tumor", "Recall Blood", "Recall Dura", "F1 Healthy", "F1 Tumor", "F1 Blood", "F1 Dura", "AUC Healthy", "AUC Tumor", "AUC Blood", "AUC Dura"]

    fold_tables = [PrettyTable() for _ in range(nfolds)]
    for f, table_fold in enumerate(fold_tables):
        table_fold.field_names = ["Dataset", "Model", f'F{f} Kappa score', f'F{f} Precision', f'F{f} Recall', f'F{f} F1', f'F{f} OACC', f'F{f} ROCAUC', "", f'F{f} ACC Healthy', f'F{f} ACC Tumor', f'F{f} ACC Blood', f'F{f} ACC Dura', f'F{f} Precision Healthy', f'F{f} Precision Tumor' , f'F{f} Precision Blood', f'F{f} Precision Dura', f'F{f} Recall Healthy', f'F{f} Recall Tumor', f'F{f} Recall Blood', f'F{f} Recall Dura', f'F{f} F1 Healthy', f'F{f} F1 Tumor', f'F{f} F1 Blood', f'F{f} F1 Dura', f'F{f} AUC Healthy', f'F{f} AUC Tumor', f'F{f} AUC Blood', f'F{f} AUC Dura']
    
    models_metrics = {}
    for experiment in tqdm(experiments, desc="Working on experiments data"):
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        for run in runs:
            db_name = run.data.params.get("db_name", None)
            job_name = run.data.params.get("job_name", None)
            seed = run.data.params.get("seed", None)

            artifact_uri = run.info.artifact_uri
            metrics = load_metrics_from_run(artifact_uri)
        
            if job_name is not None:
                if seed is not None:
                    if job_name not in models_metrics:
                        models_metrics[job_name] = {}
                    if seed not in models_metrics[job_name]:
                        models_metrics[job_name][seed] = metrics
        
            gm = global_metrics(models_metrics, nfolds)
            fm = fold_metrics(models_metrics, nfolds)

            for job_name, gm in gm.items():
                table_global.add_row([
                    db_name, job_name,
                    f"{gm.get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {gm.get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{gm.get('precision', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {gm.get('precision', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{gm.get('recall', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {gm.get('recall', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{gm.get('f1score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {gm.get('f1score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{gm.get('oacc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {gm.get('oacc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    f"{gm.get('rocauc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {gm.get('rocauc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                    "",
                    f"{gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {gm.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {gm.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {gm.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {gm.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                    f"{gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                    f"{gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                    f"{gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                    f"{gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {gm.get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}"
                ])

            for job_name, fm_dic in fm.items():
                for fold in fm_dic.keys():
                    fold_tables[int(fold)].add_row([
                        db_name, job_name,
                        f"{fm_dic[fold].get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {fm_dic[fold].get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                        f"{fm_dic[fold].get('precision', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {fm_dic[fold].get('precision', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                        f"{fm_dic[fold].get('recall', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {fm_dic[fold].get('recall', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                        f"{fm_dic[fold].get('f1score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {fm_dic[fold].get('f1score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                        f"{fm_dic[fold].get('oacc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {fm_dic[fold].get('oacc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                        f"{fm_dic[fold].get('rocauc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {fm_dic[fold].get('rocauc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
                        "",
                        f"{fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                        f"{fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                        f"{fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                        f"{fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {fm_dic[fold].get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                        f"{fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                        f"{fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                        f"{fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                        f"{fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {fm_dic[fold].get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                        f"{fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                        f"{fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                        f"{fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                        f"{fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {fm_dic[fold].get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                        f"{fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                        f"{fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                        f"{fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                        f"{fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {fm_dic[fold].get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}",
                        f"{fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][0]:.2f} ± {fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][0]:.2f}",
                        f"{fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][1]:.2f} ± {fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][1]:.2f}",
                        f"{fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][2]:.2f} ± {fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][2]:.2f}",
                        f"{fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean'][3]:.2f} ± {fm_dic[fold].get('auc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev'][3]:.2f}"
                    ])
                        

    #sort according to db_name
    table_data_global = sorted([row for row in table_global if row[0] is not None], key=lambda x: x[0])

    table_data_fold = []
    for f, table_fold in enumerate(fold_tables):
        table_data_fold.append(sorted([row for row in table_fold if row[0] is not None], key=lambda x: x[0]))


    ## PRINT TABLES
    file = open(args.fileToSave, "w")
    file.write(table_data_global.get_string())
    file.write("\n\n")
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
