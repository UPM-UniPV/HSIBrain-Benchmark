import mlflow
import os

import subprocess

tracking_uri = "file://" + os.path.abspath("mlruns_final")
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.tracking.MlflowClient()


def run_all_inferences():
    experiments = client.search_experiments()
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)

        for run in runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", run_id)

            if "MamTrans" in run_name:
                env_name = "MambaEnv"
            else:
                env_name = "ModExperiments"

            db_name = run.data.params.get("db_name", "")
            if db_name == "LP":
                data_path = "/home/ragusa/HSIBrain/datasets/LP/hsi/"
                gt_path = "/home/ragusa/HSIBrain/datasets/LP/gt/"
            else:
                data_path = "/home/ragusa/HSIBrain/datasets/Madrid/hsi/"
                gt_path = "/home/ragusa/HSIBrain/datasets/LP/gt/"

            try:
                command = f'conda run -n {env_name} python3 run_with_submitit.py --inference --run-id {run_id} --ngpus 1 --nodes 1 --db-name {db_name} --data-path {data_path} --gt-path {gt_path} --job-name "inference_{run_id}"'
                subprocess.run(command, shell=True, check=True)

                print(f"\033[32mInference for {run_id} completed\033[0m")
            except subprocess.CalledProcessError as e:
                print(f"\033[31mError processing run {run_id}: {e}\033[0m")


if __name__ == "__main__":
    run_all_inferences()
