from abc import ABC, abstractmethod


class Task(ABC):
    """
    Abstract class for all tasks
    """

    @abstractmethod
    def __init__(self) -> None:
        self.name = ""

    @abstractmethod
    def execute(self):
        """
        Execute the task and return the result
        """
