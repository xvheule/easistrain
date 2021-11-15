import h5py
import numpy as np
import silx.math.fit
import silx.math.fit.peaks
import scipy.optimize
from typing import Sequence, Union

from easistrain.EDD.utils import (
    calcBackground,
    gaussEstimation,
    run_from_cli,
    splitPseudoVoigt,
)


def get_output_group_name(dataset: str, n_scan_h: str, n_scan_v: str) -> str:
    return "fit" + "_" + str(dataset) + "_" + str(n_scan_h) + "_" + str(n_scan_v)


def guessParameters(xData, yData, counterOfBoxes, nbPeaksInBoxes):
    p0Guess = np.zeros(3 * nbPeaksInBoxes[counterOfBoxes], float)
    fwhmGuess = silx.math.fit.peaks.guess_fwhm(yData)
    peaksGuess = silx.math.fit.peaks.peak_search(
        yData,
        fwhmGuess,
        sensitivity=2.5,
        begin_index=None,
        end_index=None,
        debug=False,
        relevance_info=False,
    )  ## index of the peak with peak relevance
    # print(peaksGuess)
    if (
        np.size(peaksGuess) > nbPeaksInBoxes[counterOfBoxes]
    ):  ## case if more peaks than expected are detected
        peaksGuess = silx.math.fit.peaks.peak_search(
            yData,
            fwhmGuess,
            sensitivity=1,
            begin_index=None,
            end_index=None,
            debug=False,
            relevance_info=True,
        )  ## index of the peak with peak relevance
        peaksGuessArray = np.asarray(peaksGuess)
        orderedIndex = np.argsort(peaksGuessArray[:, 1])[
            -nbPeaksInBoxes[counterOfBoxes] :
        ]
        peaksGuess = peaksGuessArray[orderedIndex[:], 0]  ## peaks indices
    if (
        np.size(peaksGuess) < nbPeaksInBoxes[counterOfBoxes]
    ):  ## case if less peaks than expected are detected
        peaksGuess = silx.math.fit.peaks.peak_search(
            yData,
            fwhmGuess,
            sensitivity=1,
            begin_index=None,
            end_index=None,
            debug=False,
            relevance_info=True,
        )  ## index of the peak with peak relevance
        peaksGuessArray = np.asarray(peaksGuess)
        orderedIndex = np.argsort(peaksGuessArray[:, 1])[
            -nbPeaksInBoxes[counterOfBoxes] :
        ]
        peaksGuess = peaksGuessArray[orderedIndex[:], 0]  ## peaks indices
    # print(peaksGuess)
    for ipar in range(nbPeaksInBoxes[counterOfBoxes]):
        p0Guess[3 * ipar] = yData[int(peaksGuess[ipar])]
        p0Guess[3 * ipar + 1] = xData[int(peaksGuess[ipar])]
        p0Guess[3 * ipar + 2] = fwhmGuess
    firstGuess, covGuess = scipy.optimize.curve_fit(
        gaussEstimation,
        xData,
        yData,
        p0Guess,
    )
    # print(firstGuess)
    # print(peaksGuess)
    return firstGuess, peaksGuess


def calibEdd(
    fileRead: str,
    fileSave: str,
    sample: str,
    dataset: Union[str, int],
    scanNumberHorizontalDetector: Union[str, int],
    scanNumberVerticalDetector: Union[str, int],
    nameHorizontalDetector: str,
    nameVerticalDetector: str,
    numberOfBoxes: int,
    nbPeaksInBoxes: Sequence[int],
    rangeFit: Sequence[int],
    sourceCalibrantFile: str,
):
    """Main function."""

    with h5py.File(fileRead, "r") as h5Read:  ## Read the h5 file of raw data
        patternHorizontalDetector = h5Read[
            sample
            + "_"
            + str(dataset)
            + "_"
            + str(scanNumberHorizontalDetector)
            + ".1/measurement/"
            + nameHorizontalDetector
        ][
            ()
        ]  ## calibration pattern of horizontal detector
        patternVerticalDetector = h5Read[
            sample
            + "_"
            + str(dataset)
            + "_"
            + str(scanNumberVerticalDetector)
            + ".1/measurement/"
            + nameVerticalDetector
        ][
            ()
        ]  ## calibration pattern of vertical detector

    h5Save = h5py.File(fileSave, "a")  ## create h5 file to save in
    if "detectorCalibration" not in h5Save.keys():
        calibrationLevel1 = h5Save.create_group(
            "detectorCalibration"
        )  ## calibration group
    else:
        calibrationLevel1 = h5Save["detectorCalibration"]
    rawDataLevel1_1 = calibrationLevel1.create_group(
        "rawData"
        + "_"
        + str(dataset)
        + "_"
        + str(scanNumberHorizontalDetector)
        + "_"
        + str(scanNumberVerticalDetector)
    )  ## rawData subgroup in calibration group
    fitLevel1_2 = calibrationLevel1.create_group(
        get_output_group_name(
            dataset, scanNumberHorizontalDetector, scanNumberVerticalDetector
        )
    )  ## fit subgroup in calibration group
    fitLevel1_2.create_group("fitParams")  ## fit results group for the two detector
    fitLevel1_2.create_group(
        "curveCalibration"
    )  ## curve calibration group for the two detector
    fitLevel1_2.create_group(
        "calibCoeffs"
    )  ## calibration coefficients group for the two detector

    infoGroup = fitLevel1_2.create_group("infos")  ## infos group creation
    infoGroup.create_dataset(
        "fileRead", dtype=h5py.string_dtype(encoding="utf-8"), data=fileRead
    )  ## save path of raw data file in infos group
    infoGroup.create_dataset(
        "fileSave", dtype=h5py.string_dtype(encoding="utf-8"), data=fileSave
    )  ## save path of the file in which results will be saved in info group
    infoGroup.create_dataset(
        "sample", dtype=h5py.string_dtype(encoding="utf-8"), data=sample
    )  ## save the name of the sample in infos group
    infoGroup.create_dataset(
        "dataset", dtype=h5py.string_dtype(encoding="utf-8"), data=dataset
    )  ## save the name of dataset in infos group
    infoGroup.create_dataset(
        "scanNumberHorizontalDetector",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=str(scanNumberHorizontalDetector),
    )  ## save of the number of the scan containing the calibration pattern of the horizontal detector in infos group
    infoGroup.create_dataset(
        "scanNumberVerticalDetector",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=str(scanNumberVerticalDetector),
    )  ## save of the number of the scan containing the calibration pattern of the vertical detector in info group
    infoGroup.create_dataset(
        "nameHorizontalDetector",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=nameHorizontalDetector,
    )  ## save of the name of the horizontal detector in infos group
    infoGroup.create_dataset(
        "nameVerticalDetector",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=nameVerticalDetector,
    )  ## save of the name of the vertical detector in infos group
    infoGroup.create_dataset(
        "numberOfBoxes", dtype="int", data=numberOfBoxes
    )  ## save of the number of the boxes/widows extracted from the raw data in infos group
    infoGroup.create_dataset(
        "nbPeaksInBoxes", dtype="int", data=nbPeaksInBoxes
    )  ## save of the number of peaks per box/window in infos group
    infoGroup.create_dataset(
        "rangeFit", dtype="int", data=rangeFit
    )  ## save of the range of the fit of each box/window in infos group
    infoGroup.create_dataset(
        "sourceCalibrantFile",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=sourceCalibrantFile,
    )  ## save of the path of the calibrant File in infos group
    infoGroup.create_dataset(
        "fittingFunction",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data="asymmetric Pseudo-Voigt",
    )  ## save of the type of function used in the fitting of the peaks

    fitParamsHD = np.array(())
    fitParamsVD = np.array(())
    uncertaintyFitParamsHD = np.array(())
    uncertaintyFitParamsVD = np.array(())
    curveCalibrationHD = np.zeros((np.sum(nbPeaksInBoxes), 2), float)
    curveCalibrationVD = np.zeros((np.sum(nbPeaksInBoxes), 2), float)
    for i in range(numberOfBoxes):
        peakHorizontalDetector = np.transpose(
            (
                np.arange(rangeFit[2 * i], rangeFit[(2 * i) + 1]),
                patternHorizontalDetector[rangeFit[2 * i] : rangeFit[(2 * i) + 1]],
            )
        )  ## peak of the horizontal detector
        peakVerticalDetector = np.transpose(
            (
                np.arange(rangeFit[2 * i], rangeFit[(2 * i) + 1]),
                patternVerticalDetector[rangeFit[2 * i] : rangeFit[(2 * i) + 1]],
            )
        )  ## peak of the vertical detector
        backgroundHorizontalDetector = silx.math.fit.strip(
            data=peakHorizontalDetector[:, 1],
            w=5,
            niterations=5000,
            factor=1,
            anchors=None,
        )  ## stripped background of the horizontal detector (obtained by stripping the yData)
        backgroundVerticalDetector = silx.math.fit.strip(
            data=peakVerticalDetector[:, 1],
            w=5,
            niterations=5000,
            factor=1,
            anchors=None,
        )  ## stripped background of the vertical detector (obtained by stripping the yData)
        # print(backgroundHorizontalDetector)
        # print(backgroundVerticalDetector)
        fitLevel1_2.create_group(
            f"fitLine_{str(i)}"
        )  ## create group for each calibration peak
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "rawHorizontalDetector", dtype="float64", data=peakHorizontalDetector
        )  ## create dataset for raw data of each calibration peak
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "rawVerticalDetector", dtype="float64", data=peakVerticalDetector
        )  ## create dataset for raw data of each calibration peak

        peaksGuessHD, peaksIndexHD = guessParameters(
            peakHorizontalDetector[:, 0],
            peakHorizontalDetector[:, 1] - backgroundHorizontalDetector,
            i,
            nbPeaksInBoxes,
        )  ## guess fit parameters for HD
        # print(peaksIndexHD)
        peaksGuessVD, peaksIndexVD = guessParameters(
            peakVerticalDetector[:, 0],
            peakVerticalDetector[:, 1] - backgroundVerticalDetector,
            i,
            nbPeaksInBoxes,
        )  ## guess fit parameters for VD
        # print(peaksIndexVD)
        yCalculatedBackgroundHD, coeffBgdHD = calcBackground(
            peakHorizontalDetector[:, 0],
            peakHorizontalDetector[:, 1],
            peaksGuessHD[-1],
            peaksGuessHD[2],
            i,
            nbPeaksInBoxes,
            peaksIndexHD,
        )  ## calculated ybackground of the horizontal detector
        yCalculatedBackgroundVD, coeffBgdVD = calcBackground(
            peakVerticalDetector[:, 0],
            peakVerticalDetector[:, 1],
            peaksGuessVD[-1],
            peaksGuessVD[2],
            i,
            nbPeaksInBoxes,
            peaksIndexVD,
        )  ## calculated ybackground of the vertical detector
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "backgroundHorizontalDetector",
            dtype="float64",
            data=np.transpose((peakHorizontalDetector[:, 0], yCalculatedBackgroundHD)),
        )  ## create dataset for background of each calibration peak for HD
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "backgroundVerticalDetector",
            dtype="float64",
            data=np.transpose((peakVerticalDetector[:, 0], yCalculatedBackgroundVD)),
        )  ## create dataset for background of each calibration peak for VD
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "bgdSubsDataHorizontalDetector",
            dtype="float64",
            data=np.transpose(
                (
                    peakHorizontalDetector[:, 0],
                    peakHorizontalDetector[:, 1] - yCalculatedBackgroundHD,
                )
            ),
        )  ## create dataset for HD raw data after subst of background
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "bgdSubsDataVerticalDetector",
            dtype="float64",
            data=np.transpose(
                (
                    peakVerticalDetector[:, 0],
                    peakVerticalDetector[:, 1] - yCalculatedBackgroundVD,
                )
            ),
        )  ## create dataset for VD raw data after subst of background
        # print(peaksGuessHD)
        # print(peaksGuessVD)
        initialGuessHD = np.zeros(5 * nbPeaksInBoxes[i])
        initialGuessVD = np.zeros(5 * nbPeaksInBoxes[i])
        for n in range(nbPeaksInBoxes[i]):
            initialGuessHD[5 * n] = peaksGuessHD[3 * n]
            initialGuessHD[5 * n + 1] = peaksGuessHD[3 * n + 1]
            initialGuessHD[5 * n + 2] = peaksGuessHD[3 * n + 2]
            initialGuessHD[5 * n + 3] = peaksGuessHD[3 * n + 2]
            initialGuessHD[5 * n + 4] = 0.5
            initialGuessVD[5 * n] = peaksGuessVD[3 * n]
            initialGuessVD[5 * n + 1] = peaksGuessVD[3 * n + 1]
            initialGuessVD[5 * n + 2] = peaksGuessVD[3 * n + 2]
            initialGuessVD[5 * n + 3] = peaksGuessVD[3 * n + 2]
            initialGuessVD[5 * n + 4] = 0.5
        optimal_parametersHD, covarianceHD = scipy.optimize.curve_fit(
            f=splitPseudoVoigt,
            xdata=peakHorizontalDetector[:, 0],
            ydata=peakHorizontalDetector[:, 1] - yCalculatedBackgroundHD,
            p0=initialGuessHD,
            sigma=None,
        )  ## fit of the peak of the Horizontal detector
        optimal_parametersVD, covarianceVD = scipy.optimize.curve_fit(
            f=splitPseudoVoigt,
            xdata=peakVerticalDetector[:, 0],
            ydata=peakVerticalDetector[:, 1] - yCalculatedBackgroundVD,
            p0=initialGuessVD,
            sigma=None,
        )  ## fit of the peak of the Vertical detector
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "fitHorizontalDetector",
            dtype="float64",
            data=np.transpose(
                (
                    peakHorizontalDetector[:, 0],
                    splitPseudoVoigt(peakHorizontalDetector[:, 0], optimal_parametersHD)
                    + yCalculatedBackgroundHD,
                )
            ),
        )  ## fitted data of the horizontal detector
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "fitVerticalDetector",
            dtype="float64",
            data=np.transpose(
                (
                    peakVerticalDetector[:, 0],
                    splitPseudoVoigt(peakVerticalDetector[:, 0], optimal_parametersVD)
                    + yCalculatedBackgroundVD,
                )
            ),
        )  ## fitted data of the vertical detector
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "errorHorizontalDetector",
            dtype="float64",
            data=np.transpose(
                (
                    peakHorizontalDetector[:, 0],
                    np.absolute(
                        splitPseudoVoigt(
                            peakHorizontalDetector[:, 0], optimal_parametersHD
                        )
                        + yCalculatedBackgroundHD
                        - peakHorizontalDetector[:, 1]
                    ),
                )
            ),
        )  ## error of the horizontal detector
        fitLevel1_2[f"fitLine_{str(i)}"].create_dataset(
            "errorVerticalDetector",
            dtype="float64",
            data=np.transpose(
                (
                    peakVerticalDetector[:, 0],
                    np.absolute(
                        splitPseudoVoigt(
                            peakVerticalDetector[:, 0], optimal_parametersVD
                        )
                        + yCalculatedBackgroundVD
                        - peakVerticalDetector[:, 1]
                    ),
                )
            ),
        )  ## error of the vertical detector
        # print(f'optimal_parametersHD = {optimal_parametersHD}')
        # print(f'uncertauntyHD = {np.sqrt(np.diag(covarianceHD))}')
        # print(f'optimal_parametersVD = {optimal_parametersVD}')
        # print(f'uncertauntyVD = {np.sqrt(np.diag(covarianceVD))}')
        for n in range(nbPeaksInBoxes[i]):
            fitParamsHD = np.append(
                fitParamsHD,
                np.append(
                    optimal_parametersHD[5 * n : 5 * n + 5],
                    100
                    * np.sum(
                        np.absolute(
                            splitPseudoVoigt(
                                peakHorizontalDetector[:, 0], optimal_parametersHD
                            )
                            + backgroundHorizontalDetector
                            - peakHorizontalDetector[:, 1]
                        )
                    )
                    / np.sum(peakHorizontalDetector[:, 1]),
                ),
                axis=0,
            )  ##
            fitParamsVD = np.append(
                fitParamsVD,
                np.append(
                    optimal_parametersVD[5 * n : 5 * n + 5],
                    100
                    * np.sum(
                        np.absolute(
                            splitPseudoVoigt(
                                peakVerticalDetector[:, 0], optimal_parametersVD
                            )
                            + backgroundVerticalDetector
                            - peakVerticalDetector[:, 1]
                        )
                    )
                    / np.sum(peakVerticalDetector[:, 1]),
                ),
                axis=0,
            )  ##
            uncertaintyFitParamsHD = np.append(
                uncertaintyFitParamsHD,
                np.sqrt(np.diag(covarianceHD))[5 * n : 5 * n + 5],
                axis=0,
            )  ##
            uncertaintyFitParamsVD = np.append(
                uncertaintyFitParamsVD,
                np.sqrt(np.diag(covarianceVD))[5 * n : 5 * n + 5],
                axis=0,
            )  ##
    rawDataLevel1_1.create_dataset(
        "horizontalDetector", dtype="float64", data=patternHorizontalDetector
    )  ## save raw data of the horizontal detector
    rawDataLevel1_1.create_dataset(
        "verticalDetector", dtype="float64", data=patternVerticalDetector
    )  ## save raw data of the vertical detector
    fitLevel1_2["fitParams"].create_dataset(
        "fitParamsHD",
        dtype="float64",
        data=np.reshape(fitParamsHD, (int(np.size(fitParamsHD) / 6), 6)),
    )  ## save parameters of the fit of HD
    fitLevel1_2["fitParams"].create_dataset(
        "fitParamsVD",
        dtype="float64",
        data=np.reshape(fitParamsVD, (int(np.size(fitParamsVD) / 6), 6)),
    )  ## save parameters of the fit of VD
    fitLevel1_2["fitParams"].create_dataset(
        "uncertaintyFitParamsHD",
        dtype="float64",
        data=np.reshape(
            uncertaintyFitParamsHD, (int(np.size(uncertaintyFitParamsHD) / 5), 5)
        ),
    )  ## save uncertainty on the parameters of the fit of HD
    fitLevel1_2["fitParams"].create_dataset(
        "uncertaintyFitParamsVD",
        dtype="float64",
        data=np.reshape(
            uncertaintyFitParamsVD, (int(np.size(uncertaintyFitParamsVD) / 5), 5)
        ),
    )  ## save uncertainty on the parameters of the fit of VD
    calibrantSource = np.loadtxt(
        sourceCalibrantFile
    )  ## open source calibration text file
    curveCalibrationHD[:, 0] = fitLevel1_2["fitParams/fitParamsHD"][:, 1]
    curveCalibrationHD[:, 1] = calibrantSource[: np.sum(nbPeaksInBoxes)]
    fitLevel1_2["curveCalibration"].create_dataset(
        "curveCalibrationHD", dtype="float64", data=curveCalibrationHD
    )  ## curve energy VS channels for horizontal detector
    curveCalibrationVD[:, 0] = fitLevel1_2["fitParams/fitParamsVD"][:, 1]
    curveCalibrationVD[:, 1] = calibrantSource[: np.sum(nbPeaksInBoxes)]
    fitLevel1_2["curveCalibration"].create_dataset(
        "curveCalibrationVD", dtype="float64", data=curveCalibrationVD
    )  ## curve energy VS channels for vertical detector
    calibCoeffsHD, covCalibCoeffsHD = np.polyfit(
        x=curveCalibrationHD[:, 0],
        y=curveCalibrationHD[:, 1],
        deg=2,
        full=False,
        cov=True,
    )  ## calibration coefficients of the horizontal detector
    calibCoeffsVD, covCalibCoeffsVD = np.polyfit(
        x=curveCalibrationVD[:, 0],
        y=curveCalibrationVD[:, 1],
        deg=2,
        full=False,
        cov=True,
    )  ## calibration coefficients of the vertical detector
    fitLevel1_2["curveCalibration"].create_dataset(
        "fitCurveCalibrationHD",
        dtype="float64",
        data=np.transpose(
            (
                curveCalibrationHD[:, 0],
                np.poly1d(calibCoeffsHD)(curveCalibrationHD[:, 0]),
            )
        ),
    )  ## fitted curve energy VS channels for horizontal detector
    fitLevel1_2["curveCalibration"].create_dataset(
        "fitCurveCalibrationVD",
        dtype="float64",
        data=np.transpose(
            (
                curveCalibrationVD[:, 0],
                np.poly1d(calibCoeffsVD)(curveCalibrationVD[:, 0]),
            )
        ),
    )  ## fitted curve energy VS channels for vertical detector
    fitLevel1_2["curveCalibration"].create_dataset(
        "errorCurveCalibrationHD",
        dtype="float64",
        data=np.transpose(
            (
                curveCalibrationHD[:, 0],
                np.abs(
                    np.poly1d(calibCoeffsHD)(curveCalibrationHD[:, 0])
                    - curveCalibrationHD[:, 1]
                ),
            )
        ),
    )  ## error between fitted and raw curve energy VS channels for horizontal detector
    fitLevel1_2["curveCalibration"].create_dataset(
        "errorCurveCalibrationVD",
        dtype="float64",
        data=np.transpose(
            (
                curveCalibrationVD[:, 0],
                np.abs(
                    np.poly1d(calibCoeffsVD)(curveCalibrationVD[:, 0])
                    - curveCalibrationVD[:, 1]
                ),
            )
        ),
    )  ## error between fitted and raw curve energy VS channels for vertical detector
    # print(f'uncertauntyCalibCoeffsHD = {np.sqrt(np.diag(covCalibCoeffsHD))}')
    # print(f'uncertauntyCalibCoeffsVD = {np.sqrt(np.diag(covCalibCoeffsVD))}')
    fitLevel1_2["calibCoeffs"].create_dataset(
        "calibCoeffsHD", dtype="float64", data=calibCoeffsHD
    )  ## save calibration coefficients of the horizontal detector
    fitLevel1_2["calibCoeffs"].create_dataset(
        "calibCoeffsVD", dtype="float64", data=calibCoeffsVD
    )  ## save calibration coefficients of the Vertical detector
    fitLevel1_2["calibCoeffs"].create_dataset(
        "uncertaintyCalibCoeffsHD",
        dtype="float64",
        data=np.sqrt(np.diag(covCalibCoeffsHD)),
    )  ## save uncertainty on calibration coefficients of the horizontal detector
    fitLevel1_2["calibCoeffs"].create_dataset(
        "uncertaintyCalibCoeffsVD",
        dtype="float64",
        data=np.sqrt(np.diag(covCalibCoeffsVD)),
    )  ## save uncertainty on calibration coefficients of the vertical detector

    h5Save.close()
    return


if __name__ == "__main__":
    run_from_cli(calibEdd)
