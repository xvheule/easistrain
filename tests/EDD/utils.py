from typing import Optional, Type
from ewokscore.task import Task


class TaskTester:
    def __init__(self):
        self.default_inputs: dict = self.generate_default_inputs()
        self.generate_input_files()
        self.task: Type[Task] = Task

    def generate_default_inputs(self) -> dict:
        return {}

    def generate_input_files(self):
        pass

    def assert_task_results(self):
        raise NotImplementedError()

    @property
    def links(self) -> Optional[dict]:
        return None
