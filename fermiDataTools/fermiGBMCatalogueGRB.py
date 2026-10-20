import warnings
from astroquery.heasarc import Heasarc
from astropy.units import UnitsWarning
from astropy.coordinates import SkyCoord
from analysisTools.GRBData import GRBData
from pathlib import Path
import pandas as pd
from astropy.table import Table

warnings.simplefilter('ignore', category=UnitsWarning)


# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()


REPO_ROOT = Path(__file__).resolve().parents[1]
GRB_CATALOGUE_DIR = REPO_ROOT / "data" / "GRBCatalogue"


# currently not used
def downloadFermiGBMCatalogue(
        outputFilename: str = "fermi_gbm_catalogue.csv"
        ) -> None:
    """
    Downloads the Fermi GBM GRB catalogue from the HEASARC database and saves it as a CSV file.

    Args:
        outputFilename (str): The name of the output CSV file. Defaults to "fermi_gbm_catalogue.csv".
    """
    catalog = Heasarc.query_region(
        position=None, 
        catalog='fermigbrst', 
        spatial='all-sky'
    )

    dataframe: pd.DataFrame = catalog.to_pandas()
    dataframe.to_csv(outputFilename, index=False)


def getFermiGRBData(
        grbName: str
        ) -> GRBData:
    """
    Retrieves the GRB data for a given GRB name from the Fermi GBM catalogue.

    Args:
        grbName (str): The name of the GRB.

    Returns:
        GRBData: An instance of the GRBData class containing the data for the specified

    Raises:
        ValueError: If no GRB data is found for the specified GRB name.
    """
    # find the position of the GRB on the sky
    positionOnSky: SkyCoord = SkyCoord.from_name(grbName)
    # query the Fermi GBM catalogue for the GRB data 
    grbDataTable: pd.DataFrame = Heasarc.query_region(
        positionOnSky,
        catalog='fermigbrst'
    ) #(can return multiple entries if more than one GRB is found at the same position)
    grbDataFrame: pd.DataFrame = grbDataTable.to_pandas()

    numberOfEntries: int = len(grbDataFrame)
    match numberOfEntries:
        case 0: # if no entries are found, raise an error
            raise ValueError(f"No GRB data found for {grbName}.")
        case 1: # if one entry is found, return the data as a GRBData instance
            grbDataRow: pd.Series = grbDataFrame.iloc[0]
        case _: # if multiple entries are found, select the one with the same name
            # there are two formats: GRBYYMMDD[A-Z] and GRBYYMMDD###
            cleanInputName: str = grbName[3:9]
            matchingEntries: pd.DataFrame = grbDataFrame[grbDataFrame['name'].str[3:9] == cleanInputName]

            if matchingEntries.empty:
                raise ValueError(f"No matching GRB data found for {grbName}.")

            # this will fail if there are two bursts on the same day in the same position, but that is VERY unlikely
            grbDataRow: pd.Series = matchingEntries.iloc[0] # select the first matching entry

    # create a GRBData instance from the data row
    instance: GRBData = GRBData()

    instance.name = grbName
    instance.triggerTime = float(grbDataRow['trigger_time'])
    instance.stopTime = float(grbDataRow['trigger_time']) + float(grbDataRow['t90'])
    instance.ra = float(grbDataRow['ra'])
    instance.dec = float(grbDataRow['dec'])
    instance.t90 = float(grbDataRow['t90'])
    instance.t90Error = float(grbDataRow['t90_error'])
    instance.observationID = grbDataRow['name'][3:]
    instance.source = "Fermi GBM"
    instance.table = grbDataTable[grbDataTable['name'] == grbDataRow['name']]
    instance.folderPath = f"{config.generalSettings.directories.dataPath}/fermiGBM/{grbName}"

    logger.info(f"Retrieved Fermi GBM data for {grbName}.")

    return instance



if __name__ == "__main__":
    logger.info("Starting Fermi GBM catalogue query.")
    testGRBData = getFermiGRBData("GRB230307A")
    print(testGRBData)