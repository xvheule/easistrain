from pathlib import Path
from typing import Union
from ewokscore.task import Task
import numpy
import h5py

from easistrain.EDD.calibrationEDD import calibEdd as calib_edd, get_output_group_name
from .utils import TaskTester


class CalibTask(
    Task,
    input_names=[
        "fileRead",
        "fileSave",
        "sample",
        "dataset",
        "scanNumberHorizontalDetector",
        "scanNumberVerticalDetector",
        "nameHorizontalDetector",
        "nameVerticalDetector",
        "numberOfBoxes",
        "nbPeaksInBoxes",
        "rangeFit",
        "sourceCalibrantFile",
    ],
    output_names=["fileSave", "outputScanName"],
):
    def run(self):
        calib_edd(**{k: self.inputs[k] for k in self.input_names()})
        self.outputs.fileSave = self.inputs.fileSave
        self.outputs.outputScanName = get_output_group_name(
            self.inputs.dataset,
            self.inputs.scanNumberHorizontalDetector,
            self.inputs.scanNumberVerticalDetector,
        )


class CalibTester(TaskTester):
    def __init__(self, tmp_path: Path):
        self.test_data_path = (
            Path(__file__).parent.parent.resolve() / "data" / "Ba_calibration_data.hdf5"
        )
        self.default_inputs = self.generate_default_inputs(tmp_path)
        self.generate_input_files(self.default_inputs)
        self.task = CalibTask

    def generate_default_inputs(self, tmp_path: Path) -> dict:
        HERE = Path(__file__).parent.resolve()

        with h5py.File(self.test_data_path, "r") as test_file:
            nb_peaks_in_boxes = test_file["infos/nbPeaksInBoxes"][()]
            fit_ranges = test_file["infos/rangeFit"][()]
        return {
            "fileRead": str(tmp_path / "input_file.h5"),
            "fileSave": str(tmp_path / "output_file.h5"),
            "sample": "sample",
            "dataset": "0000",
            "scanNumberHorizontalDetector": 2,
            "scanNumberVerticalDetector": 1,
            "nameHorizontalDetector": "horz_detector",
            "nameVerticalDetector": "vert_detector",
            "numberOfBoxes": len(nb_peaks_in_boxes),
            "nbPeaksInBoxes": nb_peaks_in_boxes,
            "rangeFit": fit_ranges,
            "sourceCalibrantFile": str(HERE.parent.parent / "Calibrants" / "BaSource"),
        }

    def generate_input_files(self, inputs: dict):
        sample, dataset = inputs["sample"], inputs["dataset"]
        n_scan_h, name_h = (
            inputs["scanNumberHorizontalDetector"],
            inputs["nameHorizontalDetector"],
        )
        n_scan_v, name_v = (
            inputs["scanNumberVerticalDetector"],
            inputs["nameVerticalDetector"],
        )
        with h5py.File(self.test_data_path, "r") as test_file:
            with h5py.File(inputs["fileRead"], "w") as h5file:
                h5file[
                    f"{sample}_{dataset}_{n_scan_h}.1/measurement/{name_h}"
                ] = test_file["horizontal/data"][()]
                h5file[
                    f"{sample}_{dataset}_{n_scan_v}.1/measurement/{name_v}"
                ] = test_file["vertical/data"][()]

        return inputs

    def assert_task_results(self):
        with h5py.File(self.default_inputs["fileSave"], "r") as h5file:
            grp_name = f'fit_{self.default_inputs["dataset"]}_{self.default_inputs["scanNumberHorizontalDetector"]}_{self.default_inputs["scanNumberVerticalDetector"]}'
            vertical_coeffs = h5file[
                f"detectorCalibration/{grp_name}/calibCoeffs/calibCoeffsVD"
            ][()]
            horizontal_coeffs = h5file[
                f"detectorCalibration/{grp_name}/calibCoeffs/calibCoeffsHD"
            ][()]

        with h5py.File(self.test_data_path, "r") as h5file:
            ref_vertical_coeffs = h5file["vertical/coeffs"][()]
            ref_horizontal_coeffs = h5file["horizontal/coeffs"][()]
            vertical_coeffs_errors = h5file["vertical/errors"][()]
            horizontal_coeffs_errors = h5file["horizontal/errors"][()]

        assert numpy.all(
            numpy.abs(vertical_coeffs - ref_vertical_coeffs) <= vertical_coeffs_errors
        )
        assert numpy.all(
            numpy.abs(horizontal_coeffs - ref_horizontal_coeffs)
            <= horizontal_coeffs_errors
        )
