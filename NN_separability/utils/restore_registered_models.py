import mlflow
import torch
import os
from main import select_model
from mlflow.models.signature import infer_signature
from main import get_args_parser

"""
The following is a script used to correct the registered model in the mlflow tracking directory.
As we found that we were registered the models at the last epoch and not the ones with the best validation loss,

Steps to successfully use this script:
1. Correct the artifact uri running the script yamlAdapt.py
2. Open the mlflow ui and delete unwanted models
3. Run this script to register the best models. For the unwanted models it will throw an error and you can ignore it.
"""

mlflow.set_tracking_uri("file://" + os.getcwd() + "/mlruns_final_complete")
client = mlflow.tracking.MlflowClient()


def parse_mlflow_params(mlflow_params):
    parser = get_args_parser()
    args = parser.parse_args([])

    for key, value in mlflow_params.items():
        if hasattr(args, key):
            param_type = type(getattr(args, key))

            # Gestisci manualmente i valori 'False' e 'True'
            if value.strip().lower() == "none":
                parsed_value = None
            elif value.strip().lower() == "false":
                parsed_value = False
            elif value.strip().lower() == "true":
                parsed_value = True
            else:
                try:
                    parsed_value = param_type(value)
                except ValueError:
                    parsed_value = value

            setattr(args, key, parsed_value)

    return args


def register_best_models():
    experiments = client.search_experiments()
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)

        for run in runs:
            run_id = run.info.run_id
            params = run.data.params
            model_type = params.get("model_type", "unknown")
            run_name = run.data.tags.get("mlflow.runName", run_id)
            model_path = f"./tmp/{model_type}_best_model_{run_name}.pth"

            if 'MamTrans' in run_name:
                try:
                    if os.path.exists(model_path):
                        print(f"Processing run {run_id} with model {model_path}")

                        args = parse_mlflow_params(params)
                        model = select_model(args)
                        model.load_state_dict(torch.load(model_path))
                        model.eval()
                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)

                        X_sample = torch.randn(
                            1, args.channels, args.patch_size, args.patch_size).to(device)
                        y_sample = model(X_sample)
                        signature = infer_signature(
                            X_sample.cpu().numpy(), y_sample.cpu().detach().numpy())

                        with mlflow.start_run(run_id=run_id):
                            mlflow.pytorch.log_model(model, artifact_path=f"{model_type}_best_model_{run_name}",
                                                     signature=signature, registered_model_name=f"best_model_{run_name}")

                        print(
                            f"\033[32mRegistered model for run {run_id} - {run_name}\033[0m")
                    else:
                        print(
                            f"\033[31mModel file not found for run {run_id} - {run_name}: {model_path}\033[0m")
                except Exception as e:
                    print(
                        f"\033[31mError processing run {run_id} - {run_name}: {e}\033[0m")
            else:
                print(
                    f"\033[31mModel {run_name} is not a MamTrans model\033[0m")
                
                


if __name__ == "__main__":
    register_best_models()
