"""
Tools for working with the Swift BAT GRB catalogue. Currently set up to use a "data" list of lists as
a catalogue, but should really be modified to use a class instead.
"""
from ast import Raise
from pathlib import Path
import pandas as pd
from analysisTools.GRBData import GRBData

REPO_ROOT = Path(__file__).resolve().parents[1]
GRB_CATALOGUE_DIR = REPO_ROOT / "data" / "GRBCatalogue"

class SwiftGRBCatalogue:
    """
    A class to represent a GRB catalogue.

    Attributes:
        filename (str): The path to the GRB catalogue file.
        data (list[list[str]]): The data from the GRB catalogue.
        columnNames (list[str]): The column names from the GRB catalogue.
    """
    def __init__(
            self,
            filename: str
            ) -> None:
        self.filename = filename
        self.data, self.columnNames = self._importData()


    def _resolveCataloguePath(
            self
            ) -> Path:
        """Resolve a catalogue file without depending on the current working directory."""
        path = Path(self.filename)
        candidates = [path]

        if not path.is_absolute():
            candidates.extend([
                REPO_ROOT / path,
                GRB_CATALOGUE_DIR / path.name,
                Path(__file__).resolve().parent / path,
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[-1]


    def _importData(
            self
            ) -> tuple[list[list[str]], list[str]]:
        """
        Imports the data from a CSV file and returns it as a list of lists along with the column names.

        Args:
            filename (str): The path to the CSV file.

        Returns:
            tuple[list[list[str]], list[str]]: A tuple containing the data and column names.
        """
        cataloguePath = self._resolveCataloguePath()
        df = pd.read_csv(cataloguePath).fillna('').convert_dtypes()
        data = df.values.tolist()
        columnNames = df.columns.tolist()
        return data, columnNames


    def getGRBData(
            self,
            grbName: str
            ) -> GRBData:
        """
        Gets the GRB data for a given GRB name.

        Args:
            grbName (str): The name of the GRB to get data for.

        Returns:
            GRBData: An instance of the GRBData class containing the data for the given GRB name.

        Raises:
            ValueError: If the GRB name is not found in the catalogue or if the RA/Dec values are invalid.
        """
        def getLineForGRB(
                GRBName: str
                ) -> list[str]:
            for row in self.data:
                if row[0] == GRBName:
                    return row
            raise ValueError(f"GRB with name {GRBName} not found.")


        def getStartStopTime(
                row: list[str]
                ) -> tuple[float, float, float]:
            try:
                triggerTime: float = float(row[2])
                t90Time: float = float(row[8])
                t90Error: float = float(row[9])
                stopTime: float = triggerTime + t90Time
                return triggerTime, stopTime, t90Error
            except ValueError:
                raise ValueError(f"Invalid time values for {row[0]}: triggerTime={row[2]}, t90Time={row[8]}, t90Error={row[9]}")


        def getAngles(
                row: list[str]
                ) -> tuple[float, float]:
            try:
                ra: float = float(row[4])
                dec: float = float(row[5])
                return ra, dec
            except ValueError:
                raise ValueError(f"Invalid RA/Dec values for {row[0]}: RA={row[4]}, Dec={row[5]}")



        instance: GRBData = GRBData()
        instance.name = grbName
        row = getLineForGRB(grbName)
        instance.triggerTime, instance.stopTime, instance.t90Error = getStartStopTime(row)
        instance.ra, instance.dec = getAngles(row)
        instance.t90 = instance.stopTime - instance.triggerTime
        instance.observationID = row[18]
        instance.source = "Swift BAT"
        return instance



if __name__ == "__main__":
    catalogue = SwiftGRBCatalogue("summary_general.csv")
    GRB080319A = catalogue.getGRBData("GRB080319A")
    print(GRB080319A)