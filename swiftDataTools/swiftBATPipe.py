"""
Complete pipe for processing Swift BAT data for a given GRB name. Steps:
1. Download the data from the swift.ac.uk archive using wget.
    Saves the data to a data/reproc/{observationID} directory. 
2. Process the data using SwiftProcessor to generate photon counts.
    These are saved to a CSV file in a data/processed/{GRBName} directory.
"""

from swiftDataTools.swiftProcessor import ProcessSwiftData
from shutil import rmtree
from shellTools.runShell import ShellRunner
from analysisTools.GRBData import GRBData
from contextlib import chdir

# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()



def downloadSwiftBATData(
        grb: GRBData
    ) -> None:
    """
    Downloads data from the swift.ac.uk archive for a given GRB. Assumes the existance of a data directory to download the data into.

    Args:
        grb (GRBData): The GRB data for which to download the data.
    """
    # Generate the wget statement
    wgetStatement = f'wget -e \'robots=off\' -nv -w 2 -nH --cut-dirs=1 -r --no-parent --reject "index.html*" https://www.swift.ac.uk/archive/reproc/{grb.observationID}/bat/'

    # change the working directory to the data folder. Returns to the original working directory after.
    with chdir("data"): 
        shell = ShellRunner()
        shell.runShellCommand(wgetStatement)
        shell.closeShell()


def processSwiftBATData(
        grb: GRBData
        ) -> None:
    """Processes the data for a given GRB name into a usable format for analysis. This includes downloading the data from the swift.ac.uk archive, unzipping the data if it is compressed, and using the HEASoft tools to process the data into light curves for different energy ranges.

    Args:
        grb (GRBData): The GRB data for which to process the data.
        energyRange (str): The energy range to be used in the light curves. This should be in the format "min-max", where min and max are the minimum and maximum energies in keV. The BAT sensor has a energy range of 15-350 keV.
        download (bool): Whether to download the data from the swift.ac.uk archive. If False, it is assumed that the data has already been downloaded and is available in the "data/reproc" directory.
        deleteOriginal (bool): Whether to delete the original data file after processing. If True, the original data file will be deleted after the processed data has been saved.

    Returns:
        None: This function does not return anything, but it will save the processed data to a file that can be used for analysis. The processed data will include light curves for the specified energy range. The file will be saved in the same directory as the original data, with the same name but with "_processed" appended to the end of the file name.
    """
    # download the data using wget
    if config.preProcessingConfig.general.downloadData:
        logger.info(f"Downloading data for {grb.name}...")
        # download the data using the wget statement
        downloadSwiftBATData(grb)
        logger.info("Download complete.")

    # process the data using the HEASoft tools to generate light curves for the specified energy range and time bin size
    data = ProcessSwiftData(grb)

    logger.info("Processing complete.")

    # delete the original data file if specified
    if config.preProcessingConfig.general.deleteWhenDone:
        logger.info(f"Deleting original data file for {grb.name}...")
        rmtree(f'data/reproc/{grb.observationID}')
        logger.info("Deletion complete.")

    return data.processedDir

if __name__ == "__main__":
    from swiftDataTools.swiftBATCatalogueGRB import SwiftGRBCatalogue
    catalogue = SwiftGRBCatalogue("summary_general.csv")
    GRB080319B = catalogue.getGRBData("GRB080319B")
    print(processSwiftBATData(GRB080319B))
