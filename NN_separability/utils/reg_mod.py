import mlflow
from mlflow.models.signature import infer_signature

import torch
import os

mlflow.set_tracking_uri("file://" + os.getcwd() + "/mlruns_final_complete")
client = mlflow.tracking.MlflowClient()

id = 'b0793ec8db1a4ef7835a3b33c3b5c6e7'
new_name = 'best_model_MamTrans-M-P11_MamTrans-Madrid-11-run-20250120_023018'

experiments = client.search_experiments()

for exp in experiments:
    runs = client.search_runs(exp.experiment_id)

    for run in runs:
        run_id = run.info.run_id
        params = run.data.params
        model_type = params.get("model_type", "unknown")
        run_name = run.data.tags.get("mlflow.runName", run_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if id in run_id:
            try:
                model = mlflow.pytorch.load_model(f"models:/best_model_MamTrans-M-P7_MamTrans-Madrid-11-run-20250120_023018/latest")  # load last registered model
                model.to(device)
                
                model.eval()

                X_sample = torch.randn(1, int(params['channels']), int(params['patch_size']), int(params['patch_size'])).to(device)
                y_sample = model(X_sample)
                signature = infer_signature(X_sample.cpu().numpy(), y_sample.cpu().detach().numpy())

                with mlflow.start_run(run_id=run_id):
                    mlflow.pytorch.log_model(model, artifact_path=f"{model_type}_best_model_{run_name}",
                                                signature=signature, registered_model_name=f"best_model_{run_name}")

                print(
                    f"\033[32mRegistered model changed for run {run_id} - {run_name}\033[0m")
            except Exception as e:
                print(
                    f"\033[31mError processing run {run_id} - {run_name}: {e}\033[0m")