import mlflow
from mlflow.tracking import MlflowClient
from prettytable import PrettyTable

from urllib.parse import urlparse
from pathlib import Path

import numpy as np

import json
import os

import math

os.environ['MLFLOW_TRACKING_URI'] = "file:///home/domenico/mlruns_final" #ADAPT ACCORDING TO YOUR MLFLOW DIRECTORY WITH RUNS RESULTS

client = MlflowClient()
experiments = client.search_experiments()

def find_json_files_in_directory(directory_uri):
    json_files = []
    directory_path = Path(urlparse(directory_uri).path)
    
    for json_file in directory_path.rglob("*.json"):
        json_files.append(str(json_file))
    
    return json_files
    
def calculate_average_and_std(directory_uri):
    json_files = find_json_files_in_directory(directory_uri)
    
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
        "f1score_perclass": []
    }
    
    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            
            metrics["kappa_score"].append(data.get("kappa_score", 0))
            metrics["precision"].append(data.get("precision", 0))
            metrics["recall"].append(data.get("recall", 0))
            metrics["f1score"].append(data.get("f1score", 0))
            metrics["oacc"].append(data.get("oacc", 0))
            metrics["rocauc"].append(data.get("rocauc", 0))
            metrics["acc_perclass"].extend(data.get("per_class_accuracy", 0))
            metrics["precision_perclass"].extend(data.get("precision_class",0))
            metrics["recall_perclass"].extend(data.get("recall_class",0))
            metrics["f1score_perclass"].extend(data.get("fscore_class",0))
    
    return metrics
    
def aggregate_metrics_by_metrics(model_metrics):
    aggregated_results = {}

    for job_name, seed_metrics in model_metrics.items():
        all_metrics = {metric: [] for metric in seed_metrics['0'][0].keys()}  # ASSUMPTION: SEED 0 IS ALWAYS PRESENT

        for seed, metrics_list in seed_metrics.items():
            for metric_name, value in metrics_list[0].items():
                all_metrics[metric_name].append(value)

        aggregated_results[job_name] = {}
        for metric_name, values in all_metrics.items():
            flattened_values = [item for sublist in values for item in sublist]
            aggregated_results[job_name][metric_name] = {
                "mean": np.mean(flattened_values)*100,
                "stddev": np.std(flattened_values)*100
            }

    return aggregated_results
    
file = open("metrics_results.txt", "w")

table_data = []


table = PrettyTable()
table.field_names = ["Dataset", "Model", 'Kappa score', 'Precision', 'Recall', 'F1', 'OACC', 'ROCAUC', "", "ACC per class", "Precision per class", "Recall per class", "F1 per class"]

model_metrics = {}

for experiment in experiments:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    for run in runs:
        print(run.info.run_name)

        db_name = run.data.params.get("db_name", None)
        job_name = run.data.params.get("job_name", None)

        artifact_uri = run.info.artifact_uri
        metrics = calculate_average_and_std(artifact_uri)

        seed = run.data.params.get("seed", None)        
        if job_name is not None:
            if seed is not None:
                # Crea una struttura di dizionari annidati per job_name e seed
                model_metrics.setdefault(job_name, {}).setdefault(seed, []).append(metrics)
    
results = aggregate_metrics_by_metrics(model_metrics)
for job_name, results in results.items():
    table.add_row([
        db_name, job_name,
        f"{results.get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('kappa_score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('precision', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('precision', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('recall', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('recall', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('f1score', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('f1score', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('oacc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('oacc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('rocauc', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('rocauc', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        "",
        f"{results.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('acc_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('precision_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('recall_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}",
        f"{results.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['mean']:.2f} ± {results.get('f1score_perclass', {'mean': math.nan, 'stddev': math.nan})['stddev']:.2f}"
    ])

table_data = sorted([row for row in table_data if row[0] is not None], key=lambda x: x[0])

file.write(table.get_string())
file.write("\n")

file.close()
