import os
from typing import List, Optional, Dict

import pandas as pd

import wandb
import click
from pathlib import Path
from tqdm import tqdm
import yaml


def start_experiment(path: Path, project: str):
    args = get_args(path)
    return wandb.init(project=project, config=args)


def get_args(path: Path, arg_name = "args.yaml") -> Optional[Dict]:
    yaml_path = path / arg_name
    with yaml_path.open("r") as stream:
        try:
            file = yaml.safe_load(stream)
            return file
        except yaml.YAMLError as exc:
            print(exc)


def log_metrics(path: Path, filename: str = "summary.csv"):
    summary = pd.read_csv(path / filename)
    for row in tqdm(summary.to_dict("records"), "Logging Epochs", leave=False):
        wandb.log(row)


def create_lock(path: Path):
    with (path / "wandb.lock").open("w+") as fp:
        fp.write("logged to wandb")
    return


def log_one_experiment(path: Path, project: str) -> None:
    run = start_experiment(path, project)
    log_metrics(path)
    run.finish()


def is_finished(experiment: Path, final_epoch: int, log_name: str = "summary.csv") -> bool:
    summary: pd.DataFrame = pd.read_csv(experiment / log_name)
    highest_epoch: int = max(summary["epoch"].to_numpy())
    return highest_epoch >= final_epoch


def fetch_all_experiments(path: Path, single_experiment: bool, overwrite: bool, project: str, final_epoch = 309, summary: str = "summary.csv") -> List[Path]:
    if single_experiment:
        return [path / project]
    else:
        candidates = [path / str(directory) / project for directory in os.listdir(path)]
        filtered = [c for c in candidates if (not (c / "wandb.lock").exists()) or overwrite]
        filtered = [c for c in filtered if (c / summary).exists()]
        filter_finished = [c for c in filtered if is_finished(c, final_epoch)]
        return filter_finished


@click.command()
@click.option("--data", "--d", help='Path to the models logs', default="./logs")
@click.option('--overwrite/--no-overwrite', default=False, help='Path to the models logs')
@click.option("--single-experiment/--all", default=False, help="Path is treated as a single experiment or as a log directory with many")
@click.option("--project", default="ImageNetSOTA", help="Name of the WANDB project")
@click.option("--apikey", default="13f4789f1fcf20a514dd3d77b099ed4746992ae3", help="Name of the WANDB project")
def main(data: str, overwrite: bool, single_experiment: bool, project: str, apikey: str) -> None:
    wandb.login(key=apikey)
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not lead to a folder")
    experiments: List[Path] = fetch_all_experiments(path, single_experiment, overwrite, project)
    for experiment in tqdm(experiments, "Logging experiments"):
        log_one_experiment(experiment, project)


if __name__ == '__main__':
    main()