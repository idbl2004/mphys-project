# to install yaml: conda install -c anaconda pyyaml

from pathlib import Path


CONFIG_FILE_PATH = Path(__file__).resolve().parent / "config.yaml"

def importConfiguration(configFilePath: str | Path = CONFIG_FILE_PATH):
    """Load the configuration from a YAML file and return it as a DictToClass object.
    
    Args:
        configFilePath (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        DictToClass: An object that allows attribute-style access to the configuration settings.

    Note:
        Example usage:
        ```
            config = importConfiguration()
            dataPath: str = config.generalSettings.directories.dataPath
        ```
    """
    import yaml

    
    class DictToClass(dict):
        def __getattr__(self, name):
            try:
                value = self[name]
                if isinstance(value, dict):
                    value = DictToClass(value)
                return value
            except KeyError:
                raise AttributeError(f"'DictToClass' object has no attribute '{name}'")


    configYAML = yaml.safe_load(open(Path(configFilePath), 'r'))
    return DictToClass(configYAML)


def getInitialBinSize(
        source: str
    ) -> float:
    """Get the initial bin size for a given source from the configuration. Seperate function to allow for future missions to be added.

    Args:
        config: The configuration object returned by importConfiguration().
        source (str): The source of the data e.g. "swift", "fermi", "svom".

    Returns:
        float: The initial bin size for the specified source.
    """
    match source:
        case "swift":
            return float(config.preProcessingConfig.swiftBATConfig.processing.initialBinSize)
        case "fermi":
            return float(config.preProcessingConfig.fermiGMBConfig.processing.initialBinSize)
        case "svom":
            return float(config.preProcessingConfig.svomConfig.processing.initialBinSize)
        case _:
            raise ValueError(f"Unknown source: {source}. Valid options are 'swift', 'fermi', 'svom'.")


config = importConfiguration()