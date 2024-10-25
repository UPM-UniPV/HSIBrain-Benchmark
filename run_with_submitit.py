"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as classification
import submitit

import warnings

warnings.simplefilter("ignore", FutureWarning)

def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit", parents=[classification_parser])

    # To use multiple nodes choose a configuration that uses the same number of gpus in each node
    parser.add_argument("--ngpus", default=4, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    
    parser.add_argument("--job-name", default="ViT", type=str, help="Job name")
    parser.add_argument("--timeout", default=5600, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="microlab", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str, help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    p = Path(f"/home/{user}/ModelExperiments/runs")
    p.mkdir(exist_ok=True)
    return p
    #raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node, #num_gpus_per_node
        tasks_per_node=num_gpus_per_node,  # one task per GPU num_gpus_per_node
        cpus_per_task=1,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_additional_parameters={"nice": '5'},
        **kwargs
    )
    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()
