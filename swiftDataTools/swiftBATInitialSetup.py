"""
Initial setup for the Swift BAT data tools.
TODO move this into a general setup script for the whole tool kit.
"""

from pathlib import Path
import requests
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRB_CATALOGUE_DIR = DATA_DIR / "GRBCatalogue"
SUMMARY_GENERAL_TXT = GRB_CATALOGUE_DIR / "summary_general.txt"
SUMMARY_GENERAL_CSV = GRB_CATALOGUE_DIR / "summary_general.csv"


def exportDataToCSV(
    filename: str,
    csvFilename: str
    ) -> None:
    """
    Exports the data from a summary_general.txt file to a CSV file.

    Args:
        filename (str): The path to the summary_general.txt file.
        csvFilename (str): The path to the output CSV file.
    """
    # function to write the data to a csv file
    def writeToCSV(
            data: list[list[str]],
            columnNames: list[list[str]],
            filename: str
            ) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(data, columns=columnNames[0])
        df.to_csv(filename, index=False)


    def convertSummaryGeneralToList(
            filename: str
            ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Converts the summary_general.txt file to a list of lists.

        Args:
            filename (str): The path to the summary_general.txt file.

        Returns:
            tuple[list[list[str]], list[list[str]]]: A tuple containing the data and column names.
        """
        def readFile(
            filename: str
            ) -> tuple[list[str], list[str], list[str]]:
            with open(filename, 'r') as f:
                lines: list[str] = f.readlines()

            discriptionLines: list[str] = []
            dataLines: list[str] = []
            # seperate the file discrption lines from the data lines
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    discriptionLines.append(line)
                else:
                    dataLines.append(line)

            # the last line of the discription lines is the column names
            columnNames: list[str] = [discriptionLines.pop(-1).strip("## ")]
            return discriptionLines, dataLines, columnNames


        # function to clean the data lines
        def cleanData(
            dataLines: list[str]
            ) -> list[list[str]]:
            cleanedData: list[list[str]] = []
            for line in dataLines:
                line = line.split("|")
                line = [field.strip() for field in line]
                cleanedData.append(line)
            return cleanedData


        def getObservationID(
            grbName: str,
            data: list[list[str]], 
            isTrigID: bool = False
            ) -> str:
            """
            Gets the observation ID for a given GRB name or trigger ID.

            Args:
                grbName (str): The name of the GRB or trigger ID to get the observation ID for.
                data (list[list[str]]): The data containing the GRB information.
                isTrigID (bool, optional): Whether the provided name is a trigger ID. Defaults to False.

            Raises:
                ValueError: If the observation ID cannot be found or if the input types are incorrect.

            Returns:
                str: The observation ID for the given GRB name or trigger ID.
            """
            if isTrigID:
                trigID: str = grbName
            else:
                # find the Trig_ID for the given GRB name
                trigID: str = ""
                for row in data:
                    if row[0] == grbName:
                        trigID = row[1]
            # change the trigID to a string
            match trigID:
                case int():
                    trigID = str(trigID)
                case float():
                    trigID = str(int(trigID))
                case str():
                    pass
                case _:
                    raise ValueError(f"Trig_ID is of type {type(trigID)}, expected int, float, or str.")
            # return the observation id for the given Trig_ID
            if len(trigID) < 11:
                trigID = str(trigID) + "000"
                while len(trigID) < 11:
                    trigID = "0" + trigID
            return trigID


        # function to append the observation id to the data
        def appendObservationID(
            data: list[list[str]],
            columnNames: list[list[str]]
            ) -> tuple[list[list[str]], list[list[str]]]:
            columnNames[0].append("Observation_ID")
            for row in data:
                observationID = getObservationID(row[0], data)
                row.append(observationID)
            return data, columnNames


        discriptionLines, data, columnNames = readFile(filename)
        data = cleanData(data)
        columnNames = cleanData(columnNames)
        data, columnNames = appendObservationID(data, columnNames)
        return  data, columnNames


    data, columnNames = convertSummaryGeneralToList(filename)
    writeToCSV(data, columnNames, csvFilename)



# download summary_general.txt file from the Swift BAT website and update the data in the catalogue
def updateSummaryGeneralData() -> None:
    # download the summary_general.txt file from the Swift BAT website
    url: str = "https://swift.gsfc.nasa.gov/results/batgrbcat/summary_cflux/summary_general_info/summary_general.txt"
    response = requests.get(url)
    SUMMARY_GENERAL_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_GENERAL_TXT, "wb") as f:
        f.write(response.content)
    
    # update the data in the catalogue
    exportDataToCSV(str(SUMMARY_GENERAL_TXT), str(SUMMARY_GENERAL_CSV))


# create the directory structure for the data
def createDirectoryStructure() -> None:
    """
    Creates the necessary directory structure for the Swift BAT data tools.
    This function ensures that the following directories exist:
    - data/
    - data/processed/
    - data/GRBCatalogue/
    If any of these directories do not exist, they will be created.
    """
    # create the directory structure for the data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRB_CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)




# main function
def main() -> None:
    createDirectoryStructure()
    updateSummaryGeneralData()


if __name__ == "__main__":
    main()