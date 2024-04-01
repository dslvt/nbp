import argparse
import pdb
import traceback
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from hyperpyyaml.core import load_hyperpyyaml
from typing import List
from tasks.task import Task

parser = argparse.ArgumentParser(prog="recsys_research_task_manager")
parser.add_argument("-c", "--config", help="Task Manager Config")
parser.add_argument("-d", "--debug", default=False, help="Add debugger")

def execute_tasks(tasks: List[Task], run_history: pd.DataFrame) -> None:
    """
    Executes a list of tasks.
    Args:
        tasks (list): A list of tasks to be executed.
    Returns:
        None
    The function iterates through the list of tasks and attempts to execute each task.
    """
    tasks_status = []

    skip_to_task = ""
    has_to_skip = False
    run_history = run_history.sort_values(by="date")
    if run_history.shape[0] != 0:
        if run_history.iloc[-1]["status"] == "FAILED":
            skip_to_task = run_history.iloc[-1]["task_name"]
            has_to_skip = True

    for task in tasks:
        task = task["task"]
        task_execution_time = datetime.now()
        task_name = task.name

        if has_to_skip and skip_to_task != task_name:
            continue

        if has_to_skip and skip_to_task == task_name:
            has_to_skip = False

        try:
            task.execute()
            tasks_status.append((task_execution_time, task_name, "SUCCESS"))
        except Exception as _:
            logging.error(traceback.format_exc())
            tasks_status.append((task_execution_time, task_name, "FAILED"))
            break

    run_history = pd.concat(
        [
            run_history,
            pd.DataFrame(data=tasks_status, columns=["date", "task_name", "status"]),
        ]
    )

    return run_history


def main(args):
    with open(args.config) as f:
        config = load_hyperpyyaml(f)

    task_directory = Path(config["task_dir"])

    # создаем регистратор
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(task_directory / "logs.log")
    handler.setLevel(logging.DEBUG)

    # строка формата сообщения
    strfmt = "[%(asctime)s] [%(name)s] [%(levelname)s] > %(message)s"
    # строка формата времени
    datefmt = "%Y-%m-%d %H:%M:%S"
    # создаем форматтер
    formatter = logging.Formatter(fmt=strfmt, datefmt=datefmt)
    # добавляем форматтер к 'ch'
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    output_dir = task_directory / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_history = pd.read_csv(task_directory / "run_history.csv")
    run_history = execute_tasks(config["tasks"], run_history)
    run_history.to_csv(task_directory / "run_history.csv", index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        try:
            main()
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        main()