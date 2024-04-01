import logging

from tasks.task import Task


class TestTask(Task):
    def __init__(self) -> None:
        self.name = "TestTask"

    def execute(self):
        logging.info("test task complite")


class LoadFileTask(Task):
    def __init__(self) -> None:
        self.name = "LoadTestFileTask"

    def execute(self):
        logging.info("this task is broken")
        raise FileNotFoundError("file not found")


class CalculateSymbolsTask(Task):
    def execute(self):
        print("")


class SaveTask(Task):
    def execute(self):
        print("")
