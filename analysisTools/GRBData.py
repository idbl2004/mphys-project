class GRBData:
    """
    A class to represent a GRB data entry.
    * For Swift GRB data, use SwiftGRBCatalogue.getGRBData(grbName) from swiftBATCatalogueGRB to get an instance of this class for a given GRB name.
    * For Fermi GRB data, use FermiGBMCatalogue.getGRBData(grbName) from fermiGBMCatalogueGRB to get an instance of this class for a given GRB name.

    Attributes:
        name (str): The name of the GRB.
        triggerTime (float): The trigger time of the GRB.
        stopTime (float): The stop time of the GRB.
        ra (float): The right ascension of the GRB.
        dec (float): The declination of the GRB.
        t90 (float): The T90 duration of the GRB.
        t90Error (float): The error in the T90 duration of the GRB.
        observationID (str): The observation ID of the GRB.
    """
    name: str
    triggerTime: float
    stopTime: float
    ra: float
    dec: float
    t90: float
    t90Error: float
    observationID: str
    source: str


    def __repr__(self) -> str:
        attributes = [
            f"{key}={value!r}" 
            for key, value in self.__dict__.items()
        ]
        
        class_name = self.__class__.__name__
        return f"{class_name}(\n{'\n'.join(attributes)}\n)"