from pathlib import Path
from typing import Union
import h5py
from ewokscore.task import Task

from easistrain.EDD.angleCalibEDD import angleCalibrationEDD
from .utils import TaskTester


class AngleCalibTask(
    Task,
    input_names=[
        "fileRead",
        "fileSave",
        "sample",
        "dataset",
        "scanNumber",
        "nameHorizontalDetector",
        "nameVerticalDetector",
        "numberOfBoxes",
        "nbPeaksInBoxes",
        "rangeFitHD",
        "rangeFitVD",
        "pathFileDetectorCalibration",
        "scanDetectorCalibration",
        "sampleCalibrantFile",
    ],
    output_names=["fileSave"],
):
    def run(self):
        print(self.inputs)
        angleCalibrationEDD(**{k: self.inputs[k] for k in self.input_names()})
        self.outputs.fileSave = self.inputs.fileSave


class AngleCalibTester(TaskTester):
    def __init__(self, tmp_path: Path):
        self.test_data_path = (
            Path(__file__).parent.parent.resolve()
            / "data"
            / "TiC_angle_calib_data.hdf5"
        )

        self.default_inputs = self.generate_default_inputs(tmp_path)
        self.generate_input_files(self.default_inputs)
        self.task = AngleCalibTask

    def generate_default_inputs(self, tmp_path: Path) -> dict:
        HERE = Path(__file__).parent.resolve()

        with h5py.File(self.test_data_path, "r") as test_file:
            nb_peaks_in_boxes = test_file["infos/nbPeaksInBoxes"][()]
            fit_ranges_h = test_file["infos/rangeFitHD"][()]
            fit_ranges_v = test_file["infos/rangeFitVD"][()]
        return {
            "fileRead": str(tmp_path / "input_files.h5"),
            "fileSave": str(tmp_path / "output_files.h5"),
            "sample": "sample",
            "dataset": "0000",
            "scanNumber": 2,
            "nameHorizontalDetector": "horz_detector",
            "nameVerticalDetector": "vert_detector",
            "numberOfBoxes": len(nb_peaks_in_boxes),
            "nbPeaksInBoxes": nb_peaks_in_boxes,
            "rangeFitHD": fit_ranges_h,
            "rangeFitVD": fit_ranges_v,
            "sampleCalibrantFile": str(HERE.parent.parent / "Calibrants" / "TiC.d"),
        }

    def generate_input_files(self, inputs: dict):
        sample, dataset = inputs["sample"], inputs["dataset"]
        n_scan = inputs["scanNumber"]
        name_h, name_v = (
            inputs["nameHorizontalDetector"],
            inputs["nameVerticalDetector"],
        )
        with h5py.File(self.test_data_path, "r") as data_file:
            with h5py.File(inputs["fileRead"], "w") as h5file:
                h5file[
                    f"{sample}_{dataset}_{n_scan}.1/measurement/{name_h}"
                ] = data_file["horizontal/data"][()]
                h5file[
                    f"{sample}_{dataset}_{n_scan}.1/measurement/{name_v}"
                ] = data_file["vertical/data"][()]

    def assert_task_results(self):
        with h5py.File(self.test_data_path, "r") as h5file:
            ref_vertical_angle = h5file["vertical/angle"][()]
            ref_horizontal_angle = h5file["horizontal/angle"][()]
            vertical_angle_error = h5file["vertical/error"][()]
            horizontal_angle_error = h5file["horizontal/error"][()]

        with h5py.File(self.default_inputs["fileSave"], "r") as h5file:
            grp_name = f'fit_{self.default_inputs["dataset"]}_{self.default_inputs["scanNumber"]}'
            vertical_angle = h5file[
                f"angleCalibration/{grp_name}/calibratedAngle/calibratedAngleVD"
            ][()]
            horizontal_angle = h5file[
                f"angleCalibration/{grp_name}/calibratedAngle/calibratedAngleHD"
            ][()]

        assert abs(vertical_angle - ref_vertical_angle) <= vertical_angle_error
        assert abs(horizontal_angle - ref_horizontal_angle) <= horizontal_angle_error

    @property
    def links(self):
        return {
            "source": "calib",
            "data_mapping": [
                {
                    "source_output": "fileSave",
                    "target_input": "pathFileDetectorCalibration",
                },
                {
                    "source_output": "outputScanName",
                    "target_input": "scanDetectorCalibration",
                },
            ],
        }
