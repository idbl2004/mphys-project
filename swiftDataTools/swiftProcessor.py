"""
Processes Swift BAT data for a given GRB. Currently has some work arounds to handle CALDB and PERL
issues. I am unsure if these issues are due to my laptop or if they are a general issue. 
TODO investigate this.
"""


from os import environ, makedirs
from subprocess import run
from heasoftpy import Config, bateconvert, batmaskwtevt, batbinevt
from analysisTools.GRBData import GRBData
from swiftDataTools.swiftSpectralTools import SpectralProcessor
from pathlib import Path
from astropy.io import fits
import numpy as np


# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()



class ProcessSwiftData:
    """Class to process Swift BAT data for a given GRB. This class handles the processing of Swift BAT data for a given GRB, including checking and applying gain correction, checking and applying mask weighting, extracting light curves for different time periods, and converting the light curve data into CSV files. It has the ability to process a custom time range, however this will require the user to change settings in the config.yaml file. """
    def __init__(
            self,
            grb: GRBData
            )-> None:
        """Constructor for `ProcessSwiftData` class. Runs a complete pipeline to convert data from Swift BAT raw data to a standardised CSV file. Requires the data to exist in `/data/reproc/{observationID}/bat/`

        Args:
            grb (GRBData): The GRB data for which to process the data. This should be an instance of the `GRBData` class.
        Raises:
            EnvironmentError: if the HEADAS environment variable is not set. This is required for the HEASoft tools to function properly. The user should ensure that they have initialized HEASoft before running this code.
            FileNotFoundError: if the required data files are not found in the specified directory.
            ValueError: if the provided GRB name is invalid or not found in the dataset.
            ValueError: if the provided energy bins are invalid or not supported.

        Returns:
            None: Output is saved to a CSV file in the `data/processed/{GRBName}` directory.
        """
        # set the environment variables for HEASoft and CALDB
        self.__setENV()

        self.GRBName = grb.name
        self.energyBins = config.preProcessingConfig.swiftBATConfig.download.energyRange
        self.__triggerID: str = grb.observationID
        self.__rightAscension, self.__declination = grb.ra, grb.dec
        self.__startTime, self.__stopTime, _ = grb.triggerTime, grb.stopTime, grb.t90Error

        # set up the paths to the data directories and files
        self.__setPaths()

        # process the event file into a light curve
        self.__processToLightCurve()

        # load the light curves into memory
        self.__loadLightCurves()        

        # generate the spectrum and response matrix for the burst period
        self.__spectralProcessor: SpectralProcessor = SpectralProcessor(
            batPath=str(self.__triggerDir),
            startTime=self.__startTime,
            stopTime=self.__stopTime,
            triggerID=self.__triggerID,
            outputDir=str(self.__eventDir)
        )

        # might as well calculate the Epeak while the spectrum is being generated.
        # analyze the spectrum using XSPEC to find the Epeak WRONG! this is not trivial and needs its
        # own class to handle the XSPEC analysis.

        # convert the light curve counts to photons
        self.__processToPhotons()


    def __setENV(
            self
        )-> None:
        """Sets the environment variables for HEASoft and CALDB."""
        # Check if the HEADAS environment variable is set. If it is not set, raise an error message and exit the program. This is important because the HEASoft tools require the HEADAS environment variable to be set in order to function properly. If the variable is not set, the program will not be able to find the necessary tools and will fail to run. By checking for the variable at the beginning of the program, we can ensure that the user is aware of the issue and can take steps to fix it before proceeding with the data processing.
        self.__repoRoot: Path = Path(__file__).resolve().parents[1]
        self.__headasPath = environ.get("HEADAS")
        if self.__headasPath:
            # setup local separate writeable parameter directory
            self.__localPfiles: Path = self.__repoRoot / "pfiles"
            makedirs(self.__localPfiles, exist_ok=True)
            environ["PFILES"] = f"{self.__localPfiles};{Path(self.__headasPath) / 'syspfiles'}"
            
            # inject HEASoft binaries directly into Python's active execution path
            environ["PATH"] = f"{Path(self.__headasPath) / 'bin'}:{environ.get('PATH', '')}"
            environ["LD_LIBRARY_PATH"] = f"{Path(self.__headasPath) / 'lib'}:{environ.get('LD_LIBRARY_PATH', '')}"
        else:
            raise EnvironmentError("Error: HEADAS environment variable not found. Did you initialize HEASoft?")
        Config.allow_failure = False

        # set the CALDB environment variables
        self.__caldbPath = environ.get("CALDB")
        if self.__caldbPath:
            environ["CALDBCONFIG"] = str(Path(self.__caldbPath) / 'software' / 'tools' / 'caldb.config')
            environ["CALDBALIAS"] = str(Path(self.__caldbPath) / 'software' / 'tools' / 'alias_config.fits')


    def __processToLightCurve(
            self
        )-> None:
        """
        Processes the event file into a light curve using `batbinevt`. 

        Raises:
            ValueError: If the period is not 0, 1, 2, or 3, or if the period is 3 and no custom time range is provided.
        """
        # define helpers
        def extractLightCurve(
                period: int,
                customTimeRange: tuple[float, float] = None
            )-> None:
            """Does the actual extraction of the light curve.

            Args:
                period (int): Time period to extract the light curve for. 0 = pre-burst, 1 = burst, 2 = post-burst, 3 = custom. If custom is selected, the user must provide `customTimeRange`
                customTimeRange (tuple[float, float], optional): Custom time range to extract a custom light curve for. Defaults to None.
            """
            # determine the start and stop times for the light curve extraction
            match period:
                case 0: # pre-burst
                    fileName: Path = self.__preBurstLightCurvePath
                    startTime: float = self.__preBurstMidpoint - 1.0
                    stopTime: float = self.__preBurstMidpoint + 1.0
                    logger.info("Extracting pre-burst uniform light curve")
                case 1: # burst
                    fileName: Path = self.__burstLightCurvePath
                    if config.preProcessingConfig.swiftBATConfig.processing.fullBurst:
                        startTime: float = self.__startTime - 2.0
                        stopTime: float = self.__stopTime + 2.0
                        logger.info("Extracting full burst uniform light curve")
                    else:
                        startTime: float = self.__startTime - 2.0
                        stopTime: float = (
                            self.__startTime
                            + config.preProcessingConfig.swiftBATConfig.processing.sliceDuration - 2.0
                            )
                        logger.info("Extracting burst uniform light curve")
                case 2: # post-burst
                    fileName = self.__postBurstLightCurvePath
                    startTime: float = self.__postBurstMidpoint - 1.0
                    stopTime: float = self.__postBurstMidpoint + 1.0
                    logger.info("Extracting post-burst uniform light curve")
                case 3: # custom
                    if customTimeRange is None:
                        raise ValueError("Custom time range must be provided for period 3")
                    fileName: Path = self.__eventDir / "outputCustom.lc"
                    startTime: float = customTimeRange[0]
                    stopTime: float = customTimeRange[1]
                    logger.info("Extracting custom uniform light curve")
                case _:
                    raise ValueError("Invalid period. Must be 0 (pre-burst), 1 (burst), 2 (post-burst), or 3 (custom).")
            
            # run batbinevt to create the light curve (uniform bins)
            output = batbinevt(
                infile = self.__eventFilename,
                outfile = str(fileName),
                outtype = "LC",
                timedel = config.preProcessingConfig.swiftBATConfig.processing.initialBinSize,
                timebinalg = "u",
                energybins = self.energyBins,
                detmask = str(self.__hkDir / f"sw{self.__triggerID}bdqcb.hk.gz"),
                tstart = startTime,
                tstop = stopTime,
                clobber = "YES",
                outunits = "COUNTS"
            )
            logger.debug(output.stdout)


        def checkGain()-> bool:
            """Checks the gain has been applied to BAT data. Uses `fkeyprint` to check the GAINAPP and GAINMETH keywords in the event file header.

            Args:
                filename (str): The name of the file to be checked. This should be the name of the event file. Assumes that the file is in the current working directory and is still compressed.

            Returns:
                bool: True if the gain correction has been applied, False if the gain correction has not been applied
            """
            logger.info("Checking gain correction")
            # get the GAINAPP and GAINMETH keywords from the event file header
            GAINAPP: bool = self.__eventFITS[1].header['GAINAPP']
            GAINMETH: str = self.__eventFITS[1].header['GAINMETH']

            if GAINAPP and GAINMETH == "FIXEDDAC":
                logger.info("Gain correction already applied")
                return True
            else:
                logger.info("Gain correction has not been applied.")
                return False


        def correctGain()-> None:
            """Corrects the gain of BAT data.

            Currently needs to find the calibration file from the HEASARC FTP site.
            /swift/data/trend/YYYY_MM/bat/bgainoffs
            see page 31-32 of the BAT Data Analysis Guide https://swift.gsfc.nasa.gov/analysis/bat_swguide_v6_3.pdf
            Args:
                filename (str): The name of the file to be corrected. This should be the name of the event file. Assumes that the file is in the current working directory and is still compressed.

            Returns:
                None: This function does not return anything, but it will correct the gain of the BAT data in the specified file.
            """
            logger.info("Gain correction has not been applied.")
            logger.info(f"Applying gain correction to file: {self.__eventFilename}")
            # find the calibration file in the hk directory. 
            self.__calibrationFile = self.__hkDir / Path(self.__eventFilename).name.replace('bevshsp_uf.evt.gz', 'bcbo01deg00ab.fits.gz')
            if not self.__calibrationFile.exists():
                ### download the calibration file - impliment this later
                raise FileNotFoundError(f"Calibration file not found: {self.__calibrationFile}")


            # unzip the file if it is compressed. Needed for the bateconvert command to work. 
            if self.__eventFilename.endswith(".gz"):
                self.__unzipFile(self.__eventFilename)
                self.__eventFilename = self.__eventFilename.replace(".gz", "")

            # run the bateconvert command to correct the gain. 
            output = bateconvert(
                infile = self.__eventFilename,
                calfile = str(self.__calibrationFile),
                residfile = "CALDB",
                pulserfile = "CALDB",
                fitpulserfile = "CALDB",
                outfile = "NONE",
                calmode = "INDEF",
            )
            logger.debug(output.stdout)


            # zip the file back up
            self.__zipFile(self.__eventFilename)
            self.__eventFilename = self.__eventFilename + ".gz"


        def checkMask()-> bool:
            def checkMaskWeighting()-> bool:
                """Checks the mask has been applied to BAT data. Uses the output from `fkeyprint` to check the MASKAPP and MASKMETH keywords in the event file header.

                Args:
                    filename (str): The name of the file to be checked. This should be the name of the event file. Assumes that the file is in the current working directory and is still compressed.

                Returns:
                    bool: True if the mask has been applied, False if the mask has not been applied
                """
                logger.info("Checking mask weighting")
                # get the right ascension and declination from the event file header
                rightAscension: float = self.__eventFITS[1].header['BAT_RA']
                declination: float = self.__eventFITS[1].header['BAT_DEC']

                # load the absolute tolerance for position matching from the config file
                absoluteTolerance: float = float(config.preProcessingConfig.swiftBATConfig.processing.positionTolerance)

                # check if the right ascension and declination are within the absolute tolerance of the expected values
                rightAscensionMatch: bool = np.isclose(
                    rightAscension,
                    self.__rightAscension,
                    atol=absoluteTolerance
                    )
                declinationMatch: bool = np.isclose(
                    declination,
                    self.__declination,
                    atol=absoluteTolerance
                    )
                
                if rightAscensionMatch and declinationMatch:
                    logger.info("Right ascension and declination match expected values, mask weighting already applied")
                    return True
                else:
                    logger.info("Right ascension and declination do not match expected values, mask weighting has not been applied")
                    return False


            def checkMaskVersion()-> bool:
                """Checks the `batmaskwtevt` version that has been applied to the data. Uses the output from `fkeyprint` to check the MASKVER keyword in the event file header.

                Args:
                    output (FKeyPrintOutput): _description_

                Returns:
                    bool: _description_
                """
                logger.info("Checking batmaskwtevt version")
                # get the BATCREAT keyword from the event file header
                BATCREAT: str = self.__eventFITS[1].header['BATCREAT']
                version: float = float(BATCREAT.split(' ')[1].strip())
                if version >= 1.16:
                    logger.info(f"batmaskwtevt version is {version} >= 1.16")
                    return True
                else:
                    logger.info(f"batmaskwtevt version is {version} < 1.16")
                    return False


            bools: list[bool] = []
            bools.append(checkMaskWeighting())
            bools.append(checkMaskVersion())
            return all(bools)


        def applyMask()-> None:
            """Applies the mask to BAT data. Uses `batmaskwtevt` to apply the mask to the event file.

            Args:
                filename (str): The name of the file to be masked. This should be the name of the event file. Assumes that the file is in the current working directory and is still compressed.
            """
            logger.info("Mask has not been applied.")
            logger.info(f"Applying mask to file: {self.__eventFilename}")

            # find the attitude file in the aux directory.
            self.__attitudeFile = self.__auxDir / Path(self.__eventFilename).name.replace('bevshsp_uf.evt.gz', 'sat.fits.gz')
            if not self.__attitudeFile.exists():
                ### download the attitude file - impliment this later
                raise FileNotFoundError(f"Attitude file not found: {self.__attitudeFile}")
                
            # find the quality map file in the hk directory.
            self.__qualityMapFile = self.__hkDir / Path(self.__eventFilename).name.replace('bevshsp_uf.evt.gz', 'bdqcb.hk.gz')
            if not self.__qualityMapFile.exists():
                ### download the quality map file - impliment this later
                raise FileNotFoundError(f"Quality map file not found: {self.__qualityMapFile}")

            # unzip the file if it is compressed. Needed for the batmaskwtevt command to work. 
            if self.__eventFilename.endswith(".gz"):
                self.__unzipFile(self.__eventFilename)
                self.__eventFilename = self.__eventFilename.replace(".gz", "")

            # run the batmaskwtevt command to apply the mask.
            output = batmaskwtevt(
                infile = self.__eventFilename,
                attitude = str(self.__attitudeFile),
                ra = self.__rightAscension,
                dec = self.__declination,
                detmask = str(self.__qualityMapFile),
                rebalance = "YES",
                corrections = "default",
                auxfile = str(self.__eventDir / f"{self.__triggerID}bevtr.fits"),
                clobber = "YES"
            )
            logger.debug(output.stdout)

            # zip the file back up
            self.__zipFile(self.__eventFilename)
            self.__eventFilename = self.__eventFilename + ".gz"


        ##################################################
        # create the output directory if it does not exist
        logger.info(f"Processing data for {self.GRBName}...")
        self.__eventFilename: str = str(self.__eventDir / f"sw{self.__triggerID}bevshsp_uf.evt.gz")
        
        # read the event FITS file using astropy.io.fits. 
        self.__eventFITS: fits.HDUList = fits.open(self.__eventFilename)

        # check if the gain correction has been applied to the event file.
        if not checkGain():
            # correct the gain if it has not been applied.
            correctGain()
        
        # check if the mask has been applied to the event file. 
        if not checkMask():
            # apply the mask if it has not been applied.
            applyMask()

        # find the burst duration
        self.__burstDuration: float = self.__stopTime - self.__startTime
        logger.info(f"Start time: {self.__startTime} seconds")
        logger.info(f"Stop time: {self.__stopTime} seconds")
        logger.info(f"Burst duration: {self.__burstDuration} seconds")

        # get the pre and post bust midpoints 
        self.__preBurstMidpoint: float = (
            self.__startTime - self.__burstDuration
            )
        self.__postBurstMidpoint: float = (
            self.__stopTime + self.__burstDuration
            )

        # extract the light curves
        for period in range(3):
            extractLightCurve(period)


    def __unzipFile(
            self,
            filename: str
        )-> None:
        """Unzips a file using the gunzip command. This is a wrapper function for the gunzip command, which is used to unzip files that have been compressed using the gzip algorithm. The function takes in the name of the file to be unzipped, and uses the gunzip command to unzip the file. The function does not return anything, but it will print out any output from the gunzip command to the console.

        Args:
            filename (str): The name of the file to be unzipped. This should be the name of the file that has been compressed using gzip.
        """
        logger.info(f"Unzipping file: {filename}")
        run(["gunzip", filename], check=True)


    def __zipFile(
            self,
            filename: str
        )-> None:
        """Zips a file using the gzip command. This is a wrapper function for the gzip command, which is used to compress files using the gzip algorithm. The function takes in the name of the file to be zipped, and uses the gzip command to compress the file. The function does not return anything, but it will print out any output from the gzip command to the console.

        Args:
            filename (str): The name of the file to be zipped. This should be the name of the file that is to be compressed using gzip.
        """
        logger.info(f"Zipping file: {filename}")
        run(["gzip", filename, "-v"], check=True)


    def __setPaths(
            self
        )-> None:
        """Sets the paths to the data directories and files."""
        self.__triggerDir: Path = self.__repoRoot / "data" / "reproc" / self.__triggerID / "bat"
        self.__eventDir: Path = self.__triggerDir / "event"
        self.__hkDir: Path = self.__triggerDir / "hk"
        self.__auxDir: Path = self.__triggerDir / "aux"
        self.processedDir: Path = self.__repoRoot / "data" / "processed" / self.GRBName
        self.processedDir.mkdir(parents=True, exist_ok=True)
        self.__preBurstLightCurvePath: Path = self.__eventDir / "outputPreBurst.lc"
        self.__burstLightCurvePath: Path = self.__eventDir / "outputBurst.lc"
        self.__postBurstLightCurvePath: Path = self.__eventDir / "outputPostBurst.lc"


    def __loadLightCurves(
            self
        )-> None:
        """Loads a light curve from a FITS file using astropy.io.fits."""
        # load the light curve data from the output files using astropy.io.fits
        self.__preBurstLightCurve: fits.HDUList = fits.open(self.__preBurstLightCurvePath)
        self.__burstLightCurve: fits.HDUList = fits.open(self.__burstLightCurvePath)
        self.__postBurstLightCurve: fits.HDUList = fits.open(self.__postBurstLightCurvePath)


    def __processToPhotons(
            self
        )-> None:

        def convertCountsToPhotons(
                lightCurve: fits.HDUList,
                effectiveArea: float
            )-> np.ndarray:
            """
            Converts the counts in the light curve to photons using the effective area. The conversion is done by dividing the counts by the effective area.
            """
            # get the counts and errors from the light curve
            counts: np.ndarray = lightCurve[1].data['COUNTS']
            errors: np.ndarray = lightCurve[1].data['ERROR']

            # get the number of detectors from the light curve header
            numberOfDetectors: int = lightCurve[1].header['NGOODPIX']

            # convert the counts to photons using the effective area
            photons: np.ndarray = counts * numberOfDetectors / effectiveArea

            # propagate the errors using the effective area
            photonErrors: np.ndarray = errors * numberOfDetectors / effectiveArea

            return photons, photonErrors


        def writePhotonCountsToCSV(
                photonCounts: np.ndarray,
                photonErrors: np.ndarray,
                outputPath: Path
            )-> None:
            """Writes the photon counts and errors to a CSV file.

            Args:
                photonCounts (np.ndarray): The photon counts to write to the CSV file.
                photonErrors (np.ndarray): The photon errors to write to the CSV file.
                outputPath (Path): The path to the output CSV file.
            """
            # write the photon counts and errors to a CSV file
            np.savetxt(
                outputPath,
                np.column_stack((photonCounts, photonErrors)),
                delimiter=",",
                header="counts, errors",
                comments=""
            )


        # read in the response matrix using astropy.io.fits
        responseMatrixFITS: fits.HDUList = fits.open(self.__spectralProcessor.responseMatrixPath)[1]

        # get the response matrix from the FITS file. 
        responseMatrix: np.ndarray = responseMatrixFITS.data['MATRIX']

        # get the energy bins from the response matrix FITS file.
        energyLow: np.ndarray = responseMatrixFITS.data['ENERG_LO']
        energyHigh: np.ndarray = responseMatrixFITS.data['ENERG_HI']

        # find the bin centers for the energy bins
        energyBinCenters: np.ndarray = (energyLow + energyHigh) / 2.0

        # create a boolean mask to select energy bins within the specified range
        energyRange: tuple[float, float] = tuple(map(float, self.energyBins.split('-')))
        energyMask: np.ndarray = (
            energyBinCenters >= energyRange[0]) & (energyBinCenters <= energyRange[1]
            )

        # sum the response matrix rows that correspond to the selected energy bins
        responseMatrix: np.ndarray = responseMatrix[energyMask, :]
        rowSums: np.ndarray = responseMatrix.sum(axis=1)

        # calculate the effective area by averaging the row sums of the response matrix. 
        effectiveArea: float = rowSums.mean()

        # convert the bins in the light curve to photon counts using the effective area photons = counts / effective area
        self.__photonCounts, self.photonErrors = convertCountsToPhotons(
            self.__burstLightCurve,
            effectiveArea
            )
        self.__photonCountsPreBurst, self.photonErrorsPreBurst = convertCountsToPhotons(
            self.__preBurstLightCurve,
            effectiveArea
            )
        self.__photonCountsPostBurst, self.photonErrorsPostBurst = convertCountsToPhotons(
            self.__postBurstLightCurve,
            effectiveArea
            )

        # write the photon counts to a CSV file in the processed directory
        writePhotonCountsToCSV(
            self.__photonCountsPreBurst,
            self.photonErrorsPreBurst,
            self.processedDir / "photonCountsPreBurst.csv"
        )
        writePhotonCountsToCSV(
            self.__photonCounts,
            self.photonErrors,
            self.processedDir / "photonCounts.csv"
        )
        writePhotonCountsToCSV(
            self.__photonCountsPostBurst,
            self.photonErrorsPostBurst,
            self.processedDir / "photonCountsPostBurst.csv"
        )




if __name__ == "__main__":
    from swiftDataTools.swiftBATCatalogueGRB import SwiftGRBCatalogue
    catalogue: SwiftGRBCatalogue = SwiftGRBCatalogue("swiftDataTools/summary_general.csv")
    GRB080319A = catalogue.getGRBData("GRB080319A")
    data: ProcessSwiftData = ProcessSwiftData(GRB080319A)