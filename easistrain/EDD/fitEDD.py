from typing import Sequence
import numpy as np
import h5py
from easistrain.EDD.io import (
    create_info_group,
    peak_dataset_data,
    save_fit_data,
)
from easistrain.EDD.utils import fit_detector_data, run_from_cli


def fitEDD(
    fileRead: str,
    fileSave: str,
    sample: str,
    dataset: str,
    scanNumber: int,
    nameHorizontalDetector: str,
    nameVerticalDetector: str,
    positioners: Sequence[str],
    numberOfBoxes: int,
    nbPeaksInBoxes: Sequence[int],
    rangeFitHD: Sequence[int],
    rangeFitVD: Sequence[int],
):
    print(f"Fitting scan n.{scanNumber}")

    with h5py.File(fileRead, "r") as h5Read:  ## Read the h5 file of raw data
        scan_meas = h5Read.get(
            f"{sample}_{dataset}_{scanNumber}.1/measurement",
            default=None,
        )

        if (
            not isinstance(scan_meas, h5py.Group)
            or nameHorizontalDetector not in scan_meas
            or nameVerticalDetector not in scan_meas
        ):
            print("No pattern was saved in this scan")
            return

        h5Save = h5py.File(fileSave, "a")  ## create/append h5 file to save in
        scanGroup = h5Save.create_group(
            f"{sample}_{dataset}_{scanNumber}.1"
        )  ## create the group of the scan wich will contatin all the results of a scan
        positionersGroup = scanGroup.create_group(
            "positioners"
        )  ## positioners subgroup in scan group
        patternHorizontalDetector = h5Read[
            f"{sample}_{dataset}_{scanNumber}.1/measurement/{nameHorizontalDetector}"
        ][
            ()
        ]  ## pattern of horizontal detector
        patternVerticalDetector = h5Read[
            f"{sample}_{dataset}_{scanNumber}.1/measurement/{nameVerticalDetector}"
        ][
            ()
        ]  ## pattern of vertical detector
        twoD_detector_data = (
            np.ndim(patternHorizontalDetector) == 2
            or np.ndim(patternVerticalDetector) == 2
        )
        nDetectorPoints = len(patternHorizontalDetector) if twoD_detector_data else 1

        positionAngles = np.zeros((nDetectorPoints, 6), "float64")
        for i, positioner in enumerate(positioners):
            pos_data = h5Read[
                f"{sample}_{dataset}_{scanNumber}.1/instrument/positioners/{positioner}"
            ][()]
            positionersGroup.create_dataset(
                positioner,
                dtype="float64",
                data=pos_data,
            )  ## saving all the requested positioners
            if i < 6:
                positionAngles[:, i] = pos_data
            else:
                print("Too many positioners given ! Only 6 are handled for now.")

    rawDataLevel1_1 = scanGroup.create_group(
        "rawData" + "_" + str(dataset) + "_" + str(scanNumber)
    )  ## rawData subgroup in scan group
    fitGroup = scanGroup.create_group("fit")  ## fit subgroup in scan group
    tthPositionsGroup = scanGroup.create_group(
        "tthPositionsGroup"
    )  ## two theta positions subgroup in scan group
    rawDataLevel1_1.create_dataset(
        "horizontalDetector", dtype="float64", data=patternHorizontalDetector
    )  ## save raw data of the horizontal detector
    rawDataLevel1_1.create_dataset(
        "verticalDetector", dtype="float64", data=patternVerticalDetector
    )  ## save raw data of the vertical detector

    for k in range(nDetectorPoints):
        fitParams = {"horizontal": np.array(()), "vertical": np.array(())}
        uncertaintyFitParams = {
            "horizontal": np.array(()),
            "vertical": np.array(()),
        }
        pointInScan = fitGroup.create_group(
            f"{str(k).zfill(4)}"
        )  ## create a group of each pattern (point of the scan)
        fitParamsGroup = pointInScan.create_group(
            "fitParams"
        )  ## fit results group for the two detector
        for i, nb_peaks in enumerate(nbPeaksInBoxes):
            fitLine = pointInScan.create_group(
                f"fitLine_{str(i).zfill(4)}"
            )  ## create group for each range of peak(s)

            for detector in ["horizontal", "vertical"]:
                fit_min, fit_max = (
                    (rangeFitHD[2 * i], rangeFitHD[2 * i + 1])
                    if detector == "horizontal"
                    else (rangeFitVD[2 * i], rangeFitVD[2 * i + 1])
                )  # To be improved
                pattern = (
                    patternHorizontalDetector
                    if detector == "horizontal"
                    else patternVerticalDetector
                )  # To be improved
                channels = np.arange(fit_min, fit_max)
                raw_data = pattern[k, fit_min:fit_max]
                assert isinstance(raw_data, np.ndarray)
                # print(np.shape(pattern),pattern)
                (
                    background,
                    fitted_data,
                    boxFitParams,
                    uncertaintyBoxFitParams,
                ) = fit_detector_data(
                    channels=channels,
                    raw_data=raw_data,
                    nb_peaks=nb_peaks,
                    boxCounter=i,
                    scanNumber=scanNumber,
                    detectorName=detector,
                )

                save_fit_data(
                    fitLine, detector, channels, raw_data, background, fitted_data
                )

                # Accumulate fit parameters of this box
                fitParams[detector] = np.append(fitParams[detector], boxFitParams)
                uncertaintyFitParams[detector] = np.append(
                    uncertaintyFitParams[detector], uncertaintyBoxFitParams
                )
        # End of fitting procedure

        savedFitParamsHD = np.reshape(
            fitParams["horizontal"], (int(np.size(fitParams["horizontal"]) / 6), 6)
        )
        fitParamsGroup.create_dataset(
            "fitParamsHD",
            dtype="float64",
            data=savedFitParamsHD,
        )  ## save parameters of the fit of HD
        savedUncertaintyFitParamsHD = np.reshape(
            uncertaintyFitParams["horizontal"],
            (int(np.size(uncertaintyFitParams["horizontal"]) / 5), 5),
        )
        fitParamsGroup.create_dataset(
            "uncertaintyFitParamsHD",
            dtype="float64",
            data=savedUncertaintyFitParamsHD,
        )  ## save uncertainty on the parameters of the fit of HD

        savedFitParamsVD = np.reshape(
            fitParams["vertical"], (int(np.size(fitParams["vertical"]) / 6), 6)
        )
        fitParamsGroup.create_dataset(
            "fitParamsVD",
            dtype="float64",
            data=savedFitParamsVD,
        )  ## save parameters of the fit of VD
        savedUncertaintyFitParamsVD = np.reshape(
            uncertaintyFitParams["vertical"],
            (int(np.size(uncertaintyFitParams["vertical"]) / 5), 5),
        )
        fitParamsGroup.create_dataset(
            "uncertaintyFitParamsVD",
            dtype="float64",
            data=savedUncertaintyFitParamsVD,
        )  ## save uncertainty on the parameters of the fit of VD
        for peakNumber in range(np.sum(nbPeaksInBoxes)):
            if f"peak_{str(peakNumber).zfill(4)}" not in tthPositionsGroup.keys():
                peakDataset = tthPositionsGroup.create_dataset(
                    f"peak_{str(peakNumber).zfill(4)}",
                    dtype="float64",
                    data=np.zeros((2, 13), "float64"),
                )  ## create a dataset for each peak in tthPositionGroup
                uncertaintyPeakDataset = tthPositionsGroup.create_dataset(
                    f"uncertaintyPeak_{str(peakNumber).zfill(4)}",
                    dtype="float64",
                    data=np.zeros((2, 13), "float64"),
                )  ## create a dataset for uncertainty for each peak in tthPositionGroup
            else:
                peakDataset = tthPositionsGroup[f"peak_{str(peakNumber).zfill(4)}"]
                assert isinstance(peakDataset, h5py.Dataset)
                uncertaintyPeakDataset = tthPositionsGroup[
                    f"uncertaintyPeak_{str(peakNumber).zfill(4)}"
                ]
                assert isinstance(uncertaintyPeakDataset, h5py.Dataset)
            peakDataset[0] = peak_dataset_data(
                positionAngles, savedFitParamsHD[peakNumber], -90
            )
            peakDataset[1] = peak_dataset_data(
                positionAngles, savedFitParamsVD[peakNumber], 0
            )
            uncertaintyPeakDataset[0] = peak_dataset_data(
                positionAngles, savedUncertaintyFitParamsHD[peakNumber], -90
            )
            uncertaintyPeakDataset[1] = peak_dataset_data(
                positionAngles, savedUncertaintyFitParamsVD[peakNumber], 0
            )
        if "infoPeak" not in tthPositionsGroup.keys():
            tthPositionsGroup.create_dataset(
                "infoPeak",
                dtype=h5py.string_dtype(encoding="utf-8"),
                data=f"{positioners}, delta, theta, position in channel, Intenstity, FWHM, shape factor, goodness factor",
            )  ## create info about dataset saved for each peak in tthPositionGroup

    create_info_group(
        scanGroup,
        fileRead,
        fileSave,
        sample,
        dataset,
        scanNumber,
        nameHorizontalDetector,
        nameVerticalDetector,
        numberOfBoxes,
        nbPeaksInBoxes,
        rangeFitHD,
        rangeFitVD,
        positioners,
    )

    h5Save.close()

    return


def fitEDD_with_scan_number_parse(**config):
    """Wrapper function to allow scanNumber to be a list or a slice."""
    n_scan_arg = config.pop("scanNumber")
    if isinstance(n_scan_arg, int):
        fitEDD(**config, scanNumber=n_scan_arg)
    elif isinstance(n_scan_arg, list):
        for i in n_scan_arg:
            fitEDD_with_scan_number_parse(**config, scanNumber=i)
    elif isinstance(n_scan_arg, str):
        if ":" in n_scan_arg:
            min_scan, max_scan = n_scan_arg.split(":")
            for i in range(int(min_scan), int(max_scan)):
                fitEDD(**config, scanNumber=i)
        else:
            fitEDD(**config, scanNumber=int(n_scan_arg))
    else:
        raise ValueError(f"Unrecognized value for scanNumber: {n_scan_arg}")


if __name__ == "__main__":
    run_from_cli(fitEDD_with_scan_number_parse)
