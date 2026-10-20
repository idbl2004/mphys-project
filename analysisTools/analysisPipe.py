from analysisTools.GRBData import GRBData
from analysisTools.haarMethods import estimateMvtUncertainty, haarPowerMod
from fermiDataTools.fermiGBMCatalogueGRB import getFermiGRBData
from fermiDataTools.fermiGBMPipe import processFermiGBMData
from fermiDataTools.fermiGRBObservation import EXPORT_BIN_SIZE_SECONDS
from swiftDataTools.swiftBATCatalogueGRB import SwiftGRBCatalogue
from swiftDataTools.swiftBATPipe import processSwiftBATData

import numpy as np

# standard logging/configuration setup
from loadConfig import config, getInitialBinSize
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()


class GRBLightCurveAnalysis:
    """
    A class to perform light curve analysis on a given GRB. This includes calculating the time-resolved MVT and Epeak using the photon counts and errors.

    Attributes:
        grb (GRBData): The GRB data for which to perform the analysis.
    """

    def __init__(
            self,
            grbName: str,
            mission: str,
            timeRange: tuple[float, float]
            ):
        """
        Initializes the GRBLightCurveAnalysis class with the given GRB name, mission, and time range.

        Args:
            grbName (str): The name of the GRB to be analyzed.
            mission (str): The mission from which the GRB data is to be obtained. This can be either "fermi", "swift" or "svom". Note: SVOM data processing is not yet implemented.
            timeRange (tuple[float, float]): The time range for the analysis.
        """
        self.grbName = grbName
        self.mission = mission
        self.timeRange = timeRange

        # perform the initial processing of the data
        self._initialProcessing()

        # read in the processed data from the CSV file
        self._readInProcessedData()

        # calculate the MVT
        self.MVT, self.MVTError, _ = estimateMvtUncertainty(
            self.counts,
            self.errors,
            haarPowerMod,
            binSizeSeconds = self._binSize,
            nRealisations = config.mvtAnalysisConfig.numberOfRealisations
        )




    def _initialProcessing(self):
        match self.mission.lower():
            case "fermi":
                self.grb: GRBData = getFermiGRBData(self.grbName)
                self._processedDir = processFermiGBMData(self.grb, timeRange=self.timeRange)
                self._binSize: float = EXPORT_BIN_SIZE_SECONDS

            case "swift":
                self._catalogue = SwiftGRBCatalogue("summary_general.csv")
                self.grb: GRBData = self._catalogue.getGRBData(self.grbName)
                self._processedDir = processSwiftBATData(self.grb)
                self._binSize: float = getInitialBinSize("swift")

            case "svom":
                raise NotImplementedError("SVOM data processing is not yet implemented.")
            case _:
                raise ValueError(f"Invalid mission: {self.mission}. Must be either 'fermi', 'swift', or 'svom'.")



    def _readInProcessedData(self):
        """
        Reads in the processed data from the CSV file and stores it in the class attributes.

        Returns:
            None: This function does not return anything, but it will store the processed data in the class attributes.
        """
        # read in the processed data from the CSV file
        processedData: np.ndarray = np.genfromtxt(
            self._processedDir,
            delimiter=','
        )
        self.counts: np.ndarray = processedData[:, 0]
        self.errors: np.ndarray = processedData[:, 1]



if __name__ == "__main__":
    grbAnalysis = GRBLightCurveAnalysis(
        grbName="GRB230307A",
        mission="fermi",
        timeRange=(-10, 10)
    )
    print(f"MVT: {grbAnalysis.MVT} ± {grbAnalysis.MVTError}")