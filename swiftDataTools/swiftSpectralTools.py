# TODO impliment the config file


from heasoftpy import batbinevt, batdrmgen, Config
from subprocess import run
from os import environ, makedirs
from os.path import join
from pathlib import Path

# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()



class SpectralProcessor:
    """
    Class to generate a spectrum and response matrix for a given burst period using Swift BAT data. This class uses the heasoftpy library to interface with the HEASoft tools, specifically batbinevt and batdrmgen, to create the necessary files for spectral analysis.
    
    Currently, there is an issue that with the CALDB environment variable not being set correctly when using the heasoftpy wrapper for batupdatephakw, so this is done using subprocess.run to run the commands in a bash shell.
    
    This class assumes that the HEADAS and CALDB environment variables are set correctly in the user's environment.
    
    It will output the spectrum and response matrix files as outputSpectrum.pha and outputResponse.rsp in the specified output directory.
    
    The class is designed to be used in a context where the Swift BAT data has already been downloaded and initial processing has been done to ensure that the event file has been energy calibrated and the detector mask has been applied.
    """
    def __init__(
            self,
            batPath: str,
            startTime: float,
            stopTime: float,
            triggerID: str,
            timeBinMethod: str = "u",
            snrThreshold: float = 5.0,
            outputDir: str = "."
        )-> None:
        """
        Constructor for the class. Outputs outputSpectrum.pha and outputResponse.rsp to the output folder.
        Can take a while to run, if run over a long burst.

        Args:
            batPath (str): The path to the BAT data directory
            startTime (float): The start time of the burst period
            stopTime (float): The stop time of the burst period
            triggerID (str): The trigger ID of the burst
            timeBinMethod (str, optional): The time binning method. Defaults to "u".
            snrThreshold (float, optional): The signal-to-noise ratio threshold. Defaults to 5.0.
            outputDir (str, optional): The directory to output the results. Defaults to ".".

        Raises:
            EnvironmentError: If the HEADAS or CALDB environment variables are not set correctly.
        """
        # initialise class variables
        self.__repoRoot = Path(__file__).resolve().parents[1]
        self.__batPath = Path(batPath).resolve()
        self.__eventDir = self.__batPath / "event"
        self.__eventFilename = self.__eventDir / f"sw{triggerID}bevshsp_uf.evt.gz"
        self.__hkDir = self.__batPath / "hk"
        self.__detMaskFilename = self.__hkDir / f"sw{triggerID}bdqcb.hk.gz"
        self.__startTime = startTime
        self.__stopTime = stopTime
        self.__triggerID = triggerID
        self.__timeBinMethod = timeBinMethod
        self.__snrThreshold = snrThreshold
        self.__outputDir = Path(outputDir).resolve()
        self.__headasPath = environ.get("HEADAS")

        # check that the HEADAS environment variable is set
        if self.__headasPath:
            # setup local separate writeable parameter directory
            local_pfiles = self.__repoRoot / "pfiles"
            makedirs(local_pfiles, exist_ok=True)
            environ["PFILES"] = f"{local_pfiles};{self.__headasPath}/syspfiles"
            
            # inject HEASoft binaries directly into Python's active execution path
            environ["PATH"] = f"{join(self.__headasPath, 'bin')}:{environ.get('PATH', '')}"
            environ["LD_LIBRARY_PATH"] = f"{join(self.__headasPath, 'lib')}:{environ.get('LD_LIBRARY_PATH', '')}"
        else:
            raise EnvironmentError("Error: HEADAS environment variable not found. Did you initialize HEASoft?")
        Config.allow_failure = False

        # check that the output directory exists, if not create it
        if not self.__outputDir:
            raise ValueError("Error: Output directory not specified.")
        makedirs(self.__outputDir, exist_ok=True)

        # set the CALDB environment variables
        self.__caldbPath = environ.get("CALDB")
        if self.__caldbPath:
            environ["CALDBCONFIG"] = join(self.__caldbPath, 'software', 'tools', 'caldb.config')
            environ["CALDBALIAS"] = join(self.__caldbPath, 'software', 'tools', 'alias_config.fits')
        else:
            raise EnvironmentError("Error: CALDB environment variable not found. Did you initialize HEASoft?")

        # generate the spectrum and response matrix for the burst period
        self.__generateSpectrum()
        self.__generateResponseMatrix()



    def __subprocessRunCommand(
            self,
            commandString: str
        )-> None:
        """Runs a command string in a bash shell using subprocess.run. This is a wrapper function for subprocess.run, which is used to run commands in a bash shell. The function takes in a command string, and uses subprocess.run to run the command in a bash shell. The function does not return anything, but it will print out any output from the command to the console.

        Args:
            commandString (str): The command string to be run. This should be a valid bash command string.
        """
        output = run(
            commandString,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )
        logger.debug(f"STDOUT Log:\n{output.stdout}")
        if output.returncode != 0:
            logger.error(f"Command failed with code {output.returncode}!")
            logger.error(f"Error Log:\n{output.stderr}")


    def __generateSpectrum(
            self
        )-> None:
        logger.info("Generating spectrum for burst period")
        # generate the output filepath for the spectrum
        self.spectrumPath = self.__outputDir / "outputSpectrum.pha"
        # run batbinevt to create the spectrum for the burst period
        output = batbinevt(
            infile = str(self.__eventFilename),
            outfile = str(self.spectrumPath),
            outtype = "PHA",
            timedel = 0.0,
            timebinalg = self.__timeBinMethod,
            tstart = self.__startTime,
            tstop = self.__stopTime,
            energybins = 'CALDB:80',
            outunits = "RATE",
            detmask = str(self.__detMaskFilename),
            snrthresh = self.__snrThreshold,
            clobber = "YES"
        )
        logger.debug(output.stdout)

        logger.info("Applying corrections to the spectrum")
        # run batupdatephakw and batphasyserr to apply corrections to the spectrum currently this is
        # done using subprocess.run to run the commands in a bash shell. this is because the heasoftpy
        # wrapper for batupdatephakw is throwing errors about the  CALDB environment variable not being
        # set, even though it is set??? TODO
        commandString = (
            f"source {self.__headasPath}/headas-init.sh && "
            f"export LHEAPERL=$(which perl) && " # 29/07 now it is complaining about the perl path??? TODO
            f"batupdatephakw {self.spectrumPath} {self.__eventDir / f'sw{self.__triggerID}bevtr.fits.gz'} clobber=YES"
        )
        self.__subprocessRunCommand(commandString)  

        # Construct the bash command string, injecting your HEASoft and CALDB setup
        commandString = (
            f"source {self.__headasPath}/headas-init.sh && "
            f"export LHEAPERL=$(which perl) && "
            f"batphasyserr {self.spectrumPath} CALDB clobber=YES"
        )
        self.__subprocessRunCommand(commandString)  


    def __generateResponseMatrix(
            self
        )-> None:
        logger.info("Generating response matrix for burst period")
        # generate the output filepath for the response matrix
        self.responseMatrixPath = self.__outputDir / "outputResponse.rsp"
        # run batdrmgen to create the response matrix for the burst period
        output = batdrmgen(
            infile = str(self.spectrumPath),
            outfile = str(self.responseMatrixPath),
            hkfile = 'NONE',
            clobber = "YES"
        )
        logger.debug(output.stdout)


if __name__ == "__main__":
    # test
    from swiftDataTools.swiftBATCatalogueGRB import getObservationID, importData, getStartStopTime


    data, _ = importData('swiftDataTools/summary_general.csv')
    GRBName = "GRB080319B"
    triggerID = getObservationID(
            GRBName,
            data,
            isTrigID=False
            )
    startTime, stopTime, _ = getStartStopTime(
        GRBName,
        data
        )

    processor = SpectralProcessor(
        batPath="data/reproc/00306757000/bat",
        startTime=startTime,
        stopTime=stopTime,
        triggerID=triggerID,
        timeBinMethod="snr",
        snrThreshold=50.0,
        outputDir="data/reproc/00306757000/bat/spectral"
    )