import argparse
import glob
import itertools
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from hyperpyyaml.core import load_hyperpyyaml

parser = argparse.ArgumentParser(prog="Datasphere Config Generator")
parser.add_argument("-c", "--config", required=True, help="Task Manager Config")
parser.add_argument(
    "-ci", "--cloud_instance", default="c1.4", help="Task Manager Config"
)


if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    args = parser.parse_args()

    with open(args.config) as f:
        config = load_hyperpyyaml(f)

    task_dir = Path(config["task_dir"])

    datasphere_config = {}
    # параметры точки входа для запуска вычислений
    datasphere_config["cmd"] = (
        f"python task_manager.py --config {str(task_dir)}/task_config.yaml"
    )

    # добавлям инфу про название и описание задания
    datasphere_config["name"] = f"{config['name']}_{config['date']}"
    datasphere_config["desc"] = ""
    if "desc" in config.keys():
        datasphere_config["desc"] = config["desc"]

    # добавляет все файлы во входные данные (это папки tasks и utils)
    script_folders = ["tasks", "utils"]
    file_paths_to_add = [
        glob.glob(f"{folder}/**/*.py", recursive=True) for folder in script_folders
    ]
    file_paths_to_add = list(itertools.chain(*file_paths_to_add))
    file_paths_to_add.append("task_manager.py")
    datasphere_config["inputs"] = file_paths_to_add

    # добавляет все файлы с папки экспа (run_history.csv и logs.log)
    files_to_add = ["logs.log", "run_history.csv", "task_config.yaml"]
    for file in files_to_add:
        file_paths_to_add.append(str(task_dir / file))

    # добавляет файлы с результатами
    tasks = config["tasks"]
    output_files = []
    for task in tasks:
        output_files.extend(task["download_files"])
    output_files.append(str(task_dir / "logs.log"))
    output_files.append(str(task_dir / "run_history.csv"))
    datasphere_config["outputs"] = output_files

    # добавляет s3 идентификаторы коннекторов
    s3_connector_ids = os.environ.get("S3_CONNECTIONS", "").split(",")
    datasphere_config["s3-mounts"] = s3_connector_ids

    # добавляет все параметры окружения из .env
    # datasphere_config["env"] = dict(dotenv_values(".env"))

    # если в папке с экспом есть requirements.txt и/или python.version, то добавляет его в сборку зависимости окружения
    datasphere_config["env"] = {}
    if os.path.isfile(task_dir / "requirements.txt"):
        datasphere_config["env"]["python"] = {
            "type": "manual",
            "requirements-file": str(task_dir / "requirements.txt"),
        }
        if os.path.isfile(task_dir / "python.version"):
            with open(task_dir / "python.version", "r") as pv:
                datasphere_config["python"]["version"] = pv.read()
    else:
        datasphere_config["python"] = "auto"

    # добавляет стандартный конфигуратор вычислительных ресурсов
    datasphere_config["cloud-instance-type"] = args.cloud_instance

    # сохраняет конфиг
    with open(task_dir / "datasphere_config.yaml", "w") as file:
        yaml.dump(datasphere_config, file)

    print(str(task_dir / "datasphere_config.yaml"))