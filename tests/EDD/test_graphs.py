from pathlib import Path
from typing import Dict
from ewokscore.utils import qualname
from ewokscore import execute_graph

from .angle_calibration import AngleCalibTester
from .calibration import CalibTester
from .utils import TaskTester


def edd_graph(testers: Dict[str, TaskTester]):
    nodes = [
        {
            "id": task_id,
            "task_type": "class",
            "task_identifier": qualname(tester.task),
            "default_inputs": [
                {"name": name, "value": value}
                for name, value in tester.default_inputs.items()
            ],
        }
        for task_id, tester in testers.items()
    ]
    links = [
        {"target": task_id, **tester.links}
        for task_id, tester in testers.items()
        if tester.links is not None
    ]
    return {"nodes": nodes, "links": links}


def test_edd_graph(tmp_path: Path):
    testers: Dict[str, TaskTester] = {
        "calib": CalibTester(tmp_path),
        "angle_calib": AngleCalibTester(tmp_path),
    }
    graph = edd_graph(testers)
    results = execute_graph(graph)
    assert len(results) == len(testers)
    for node_id, task in results.items():
        assert task.succeeded, node_id
        testers[node_id].assert_task_results()
