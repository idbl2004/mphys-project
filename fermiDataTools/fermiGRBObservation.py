"""
This will NOT run on a laptop due to the large memory requirements of the Fermi GBM data!
"""


from analysisTools.GRBData import GRBData
from gdt.core.binning.unbinned import bin_by_edges as binByEdges
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.phaii import Phaii
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
import numpy as np, gc, os, pandas as pd
from more_itertools import distinct_combinations as combinations
from analysisTools.haarMethods import estimateMvtUncertainty, haarPowerMod


# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()


# bin size of the exported photon counts CSV (100 μs)
EXPORT_BIN_SIZE_SECONDS: float = 1.0e-4

class ProcessFermiData:
    """
    Class to process Fermi GBM data for a given GRB. This class will process a given GRBData object into a CSV file with the binned photon counts. 
    
    Follows the method outlined in Bala et al. (2026)
    """
    def __init__(
            self,
            grb: GRBData,
            timeRange: tuple[float, float] = (-10, 60)
        ) -> None:
        """Constructor 

        Args:
            grb (GRBData): Instance of the GRBData class for the GRB to be processed.
        """
        self._grb: GRBData = grb
        self._detectorList: list[str] = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
        self._detectorData: dict[str, Phaii] = {}
        self._binSize: float = float(config.preProcessingConfig.fermiGMBConfig.processing.initialBinSize)
        self._timeRange: tuple[float, float] = timeRange
        self._energyRange: tuple[float, float] = (
            float(config.preProcessingConfig.fermiGMBConfig.download.energyRange.split("-")[0]),
            float(config.preProcessingConfig.fermiGMBConfig.download.energyRange.split("-")[1])
        )

        # initial read in
        self._readInTTEData()


        # find the optimal detector combination for the GRB
        self._generateDetectorCombinations()
        self._optimalDetectorCombination: tuple[str] = self._findOptimalDetectorCombination()
        logger.debug(f"Optimal detector combination: {self._optimalDetectorCombination}")

        # bin to 100 μs and export to CSV
        self.outputFilePath: str = self._exportOptimalDataToCSV(self._optimalDetectorCombination)
        logger.info(f"Exported binned photon counts to: {self.outputFilePath}")


    def _readInTTEData(
            self
        ) -> None:
        for detector in self._detectorList:
            try:
                self._detectorData[detector] = self._getSingleDetectorData(detector)
            except Exception as e:
                logger.error(f"Error processing detector {detector}: {e}")
                continue


    def _getSingleDetectorData(
            self,
            detector: str
        ) -> dict:
        """
        Retrieves raw event timestamps within energy bounds and fits a continuous background model. Discards heavy objects to free up memory.

        Args:
            detector (str): The name of the detector to process e.g. 'n0', 'n1', 'n2', etc.

        Returns:
            dict: Contains 'rawTimes' (1D array), 'backgroundModel' (GDT object) and 'initialBinWidth' (float).
        """
        # generate path for the TTE file
        filePath: str = f"{self._grb.folderPath}/current/glg_tte_{detector}_bn{self._grb.observationID}_v00.fit"

        # read and process the TTE event list within local scope
        logger.debug(f"Processing data for GRB {self._grb.name} from detector {detector} with file path {filePath}")
        tteData: GbmTte = GbmTte.open(filePath)

        # filter the TTE data by energy
        tteDataFiltered = tteData.slice_energy(self._energyRange)

        # extract the raw event arrival times (unbinned)
        extractedRawTimes: np.ndarray = np.array(tteDataFiltered.data.times)

        # create a coarse time grid for background fitting
        coarseStartTime: float = self._timeRange[0]
        coarseEndTime: float = self._timeRange[1]
        coarseGrid: np.ndarray = np.arange(coarseStartTime, coarseEndTime + self._binSize, self._binSize)

        logger.debug(f"Binning temporary data for background fit on detector {detector}")
        tempPhaii: Phaii = tteData.to_phaii(
            binByEdges,
            coarseGrid,
            time_range=self._timeRange
        )

        # fit the background model to the temporary coarse structure
        logger.debug(f"Fitting polynomial background model for detector {detector}")
        fitter = BackgroundFitter.from_phaii(tempPhaii, Polynomial) 
        fitter.fit(order=1) 

        # interpolate the fitted background model to create a continuous background model
        continuousBackgroundModel = fitter.interpolate_bins(
            tempPhaii.data.tstart,
            tempPhaii.data.tstop
        )

        # clear all references and wipe the heavy objects from RAM
        logger.debug(f"Purging GDT tracking structures from RAM for detector {detector}")
        tteData.close()
        tteDataFiltered.close()

        del tteData, tteDataFiltered, tempPhaii, fitter
        gc.collect()

        return {
            "rawTimes": extractedRawTimes,
            "backgroundModel": continuousBackgroundModel,
            "initialBinWidth": float(self._binSize)
        }


    def _generateDetectorCombinations(
            self
        ) -> None:
        """
        Generates detector combinations for the fermi mission. This will generate combinations up to a maximum specified in the config file.
        """
        maxCombinations: int = config.preProcessingConfig.fermiGMBConfig.processing.maxDetectorCombinations
        self._combinationsList: list[str] = []
        for i in range(1, maxCombinations + 1):
            self._combinationsList += list(combinations(self._detectorList, i))


    def _combineDetectorData(
            self,
            detectorCombination: tuple[str, ...]
        ) -> dict:
        """
        Dynamically bins and combines raw datasets from multiple detectors based on the current class iteration bin width.

        Args:
            detectorCombination (tuple[str, ...]): The detector combination to combine e.g. ('n0', 'n1', 'n2').
        
        Returns:
            dict: The combined data containing total counts, combined count uncertainties, summed expected background counts, and the iteration bin width.
        """
        # generate the strict, updated time boundaries based on the current bin size
        startTime: float = self._timeRange[0]
        endTime: float = self._timeRange[1]
        binEdges: np.ndarray = np.arange(startTime, endTime + self._binSize, self._binSize)
        
        # calculate bin centers and widths
        binCenters: np.ndarray = (binEdges[:-1] + binEdges[1:]) / 2.0
        binWidths: np.ndarray = binEdges[1:] - binEdges[:-1]

        # initialise blank arrays matching the active timeline resolution
        totalBins: int = len(binEdges) - 1
        combinedCounts: np.ndarray = np.zeros(totalBins, dtype=float)
        combinedBackgroundCounts: np.ndarray = np.zeros(totalBins, dtype=float)
        combinedVariance: np.ndarray = np.zeros(totalBins, dtype=float)

        # loop across all targeted detectors
        for detector in detectorCombination:
            detectorPayload = self._detectorData[detector]

            # bin the raw event timestamps into the current iteration bin size
            detectorCounts, _ = np.histogram(detectorPayload["rawTimes"], bins=binEdges)
            combinedCounts += detectorCounts

            # re-calculate individual variance
            combinedVariance += detectorCounts

            # get the coarse time midpoints
            coarseRatesObj = detectorPayload["backgroundModel"]
            
            # integrate over the specific energy window to isolate the correct channels
            lowEnergyCut, highEnergyCut = self._energyRange
            bkgdLightcurve = coarseRatesObj.integrate_energy(lowEnergyCut, highEnergyCut)
            
            # compute time midpoints
            coarseBinCenters = (bkgdLightcurve.tstart + bkgdLightcurve.tstop) / 2.0
            coarseRates = np.array(bkgdLightcurve.rates).flatten()

            # map the energy-matched background rates to the current binCenters
            detectorRates: np.ndarray = np.interp(binCenters, coarseBinCenters, coarseRates)
            
            # convert rates to expected counts
            combinedBackgroundCounts += (detectorRates * binWidths)

        # convert the total variance array back to standard deviation
        combinedUncertainties: np.ndarray = np.sqrt(combinedVariance)

        return {
            "counts": combinedCounts,
            "countUncertainties": combinedUncertainties,
            "background": combinedBackgroundCounts,
            "binWidth": float(self._binSize)
        }


    def _findPeakSNR(
            self,
            detectorCombination: tuple[str, ...],
            data: dict
        ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """
        Calculates the peak signal-to-noise ratio for a given combined detector dataset using time-resolved background count arrays.

        Args:
            detectorCombination (tuple[str, ...]): The detector combination being analyzed e.g. ('n0', 'n1', 'n2').
            data (dict): A dictionary containing 'counts', 'countUncertainties', 'background', and 'binWidth'.
        
        Returns:
            tuple: (calculatedSNR, peakCounts, expectedBackgroundInPeakBin, averageBackgroundCPS, countsArray, countUncertainties)
        """
        try:
            countsArray: np.ndarray = data["counts"]
            countUncertainties: np.ndarray = data["countUncertainties"]
            backgroundArray: np.ndarray = data["background"]
            binWidthSec: float = data["binWidth"]

            # safety validation check to avoid dividing by zero or empty datasets
            if not countsArray.any() or backgroundArray.min() <= 0:
                return 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])
            
            # calculate the SNR timeline
            snrTimeline: np.ndarray = countsArray / np.sqrt(backgroundArray)
            
            # find the peak SNR index
            peakIndex: int = int(np.argmax(snrTimeline))
            calculatedSNR: float = float(snrTimeline[peakIndex])
            peakCounts: float = float(countsArray[peakIndex])
            expectedBackgroundInPeakBin: float = float(backgroundArray[peakIndex])
            
            # reconstruct the average background counts per second (CPS) across the entire time range
            totalTimeDuration: float = len(backgroundArray) * binWidthSec
            averageBackgroundCPS: float = float(np.sum(backgroundArray) / totalTimeDuration)

            # clean up values
            peakCounts = round(peakCounts, 2)
            expectedBackgroundInPeakBin = round(expectedBackgroundInPeakBin, 2)
            averageBackgroundCPS = round(averageBackgroundCPS, 2)

            return calculatedSNR, peakCounts, expectedBackgroundInPeakBin, averageBackgroundCPS, countsArray, countUncertainties

        except Exception as errorTrace:
            logger.error(f"An error occurred during the SNR calculation for detector combination {detectorCombination}: {errorTrace}")
            return 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])


    def _findHigestSNRCombination(
            self
        ) -> tuple[tuple[str, ...], float, float, float, float, np.ndarray, np.ndarray]:
        """
        Finds the detector combination with the highest peak SNR.

        Returns:
            tuple: (bestDetectorCombination, bestSNR, bestPeakCounts, 
                    bestExpectedBackgroundInBin, bestBackgroundCPS, 
                    bestCountsArray, bestCountUncertainties)
        """
        bestSNR: float = 0.0
        bestDetectorCombination: tuple[str, ...] = ()
        bestPeakCounts: float = 0.0
        bestExpectedBackgroundInBin: float = 0.0
        bestBackgroundCPS: float = 0.0
        
        # initialise empty arrays
        bestCountsArray: np.ndarray = np.array([])
        bestCountUncertainties: np.ndarray = np.array([])

        for detectorCombination in self._combinationsList:
            combinedData: dict = self._combineDetectorData(detectorCombination)
            
            # compute the time-resolved SNR
            (calculatedSNR, 
            peakCounts, 
            expectedBackgroundInBin, 
            backgroundCPS, 
            countsArray, 
            countUncertainties) = self._findPeakSNR(detectorCombination, combinedData)

            # tracking step
            if calculatedSNR > bestSNR:
                bestSNR = calculatedSNR
                bestDetectorCombination = detectorCombination
                bestPeakCounts = peakCounts
                bestExpectedBackgroundInBin = expectedBackgroundInBin
                bestBackgroundCPS = backgroundCPS
                bestCountsArray = countsArray
                bestCountUncertainties = countUncertainties

        return (
            bestDetectorCombination, 
            bestSNR, 
            bestPeakCounts, 
            bestExpectedBackgroundInBin, 
            bestBackgroundCPS, 
            bestCountsArray, 
            bestCountUncertainties
        )


    def _findOptimalDetectorCombination(
            self
        ) -> tuple[str, ...]:
        """
        Finds the optimal detector combination for the GRB based on the highest peak SNR. Converges the bin size using the Minimum Variability Timescale (MVT). If the maxDetectorCombinations setting in the config file is large this can take a long time and require a lot of memory.

        Returns:
            tuple[str, ...]: The optimal detector combination.
        """
        currentSNRValue: float = 0.0
        currentDetectorCombination: tuple[str, ...] = None
        loopCounter: int = 0
        maxIterations: int = 15  # Safety threshold to prevent infinite loops

        # read the raw files 
        logger.info(f"Initialising raw data cache and background models for GRB {self._grb.name}...")
        self._readInTTEData() 

        while loopCounter < maxIterations:
            loopCounter += 1
            
            (bestDetectorCombination, 
            bestSNR, _, _, _, 
            countsArray, 
            countUncertainties) = self._findHigestSNRCombination()
            
            snrDifference: float = abs(bestSNR - currentSNRValue)
            logger.debug(f"Iteration {loopCounter} - Best SNR: {bestSNR} for combo: {bestDetectorCombination} with difference: {snrDifference}")
            
            # check for convergence
            if round(snrDifference, 3) < 0.001:
                logger.info(f"Converged after {loopCounter} passes.")
                logger.info(f"Final SNR: {bestSNR} for combination: {bestDetectorCombination}")
                return bestDetectorCombination
            
            # update state values
            currentSNRValue = bestSNR
            currentDetectorCombination = bestDetectorCombination
            logger.debug(f"Current state set - SNR: {currentSNRValue} for combo: {currentDetectorCombination}")

            # find the MVT
            minimumVariabilityTimescale, minimumVariabilityTimescaleUncertainty, _ = estimateMvtUncertainty(
                countsArray,
                countUncertainties,
                haarPowerMod,
                binSizeSeconds = self._binSize,
                nRealisations = config.mvtAnalysisConfig.numberOfRealisations
            )
            logger.debug(f"Calculated MVT: {minimumVariabilityTimescale} for detector combination: {currentDetectorCombination}")
            # check for invalid or zero timescale and return the current best guess if so
            if minimumVariabilityTimescale <= 0.0:
                fallbackResolution: float =  100e-6 # fallback to the instrument resolution
                logger.warning(f"haarPowerMod returned an invalid or zero timescale ({minimumVariabilityTimescale}). Applying fallback resolution floor: {fallbackResolution}")
                logger.info(f"Final SNR: {currentSNRValue} for combination: {currentDetectorCombination}")
                return currentDetectorCombination

            # set the bin size to the new MVT width
            self._binSize = minimumVariabilityTimescale
            logger.debug(f"Setting bin size to MVT: {minimumVariabilityTimescale}")

        logger.warning(f"Reached maximum iteration limit ({maxIterations}) without convergence. Returning best guess.")
        logger.info(f"Final SNR: {currentSNRValue} for combination: {currentDetectorCombination}")
        return currentDetectorCombination


    def _exportOptimalDataToCSV(
            self,
            optimalCombination: tuple[str, ...]
        ) -> str:
        """
        Takes the optimal detector combination, bins the raw timestamps to 100 microseconds, and saves the count and error arrays to a CSV file.
        
        Args:
            optimalCombination (tuple[str, ...]): The detector sequence to export e.g. ('n0', 'n1', 'n2').
            outputFileName (str): The filename for the exported CSV.
            
        Returns:
            str: Path to the generated output file.
        """
        import os
        import pandas as pd

        logger.info(f"Exporting data for combination {optimalCombination} at 100 microseconds resolution...")

        outputFileName: str = "photonCounts.csv"

        fixedBinSize: float = EXPORT_BIN_SIZE_SECONDS
        startTime: float = self._timeRange[0]
        endTime: float = self._timeRange[1]
        
        binEdges: np.ndarray = np.arange(startTime, endTime + fixedBinSize, fixedBinSize)
        
        totalBins: int = len(binEdges) - 1
        combinedCounts = np.zeros(totalBins, dtype=float)

        for detector in optimalCombination:
            detectorPayload = self._detectorData[detector]
            
            detectorCounts, _ = np.histogram(detectorPayload["rawTimes"], bins=binEdges)
            combinedCounts += detectorCounts

        combinedErrors: np.ndarray = np.sqrt(combinedCounts)

        dataFrame = pd.DataFrame({
            "counts": combinedCounts,
            "errors": combinedErrors
        })

        outputPath = os.path.join(
            config.generalSettings.directories.processedDataPath,
            self._grb.name,
            outputFileName
        )

        parentDirectory = os.path.dirname(outputPath)
        os.makedirs(parentDirectory, exist_ok=True)
        

        dataFrame.to_csv(outputPath, index=False, header=False)
        logger.info(f"Successfully generated binned output at: {outputPath}")


        return outputPath



if __name__ == "__main__":
    from fermiDataTools.fermiGBMCatalogueGRB import getFermiGRBData
    testGRBData = getFermiGRBData("GRB230307A")
    testProcessFermiData = ProcessFermiData(testGRBData)