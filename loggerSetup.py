"""
Has the following logging levels:
- DEBUG: logger.debug() - Detailed information
- INFO: logger.info() - Confirmation that things are working as expected.
- WARNING: logger.warning() - An indication that something unexpected happened.
- ERROR: logger.error() - An indication that a more serious problem occurred.
- CRITICAL: logger.critical() - An indication that a serious error occurred.


Should be initalised at the top level of the main script so that all modules can use the same logging
configuration, using:

from loggerSetup import initialiseLogging
initialiseLogging()
import logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("This is an info message.")

For submodules, use:
import logging
logger = logging.getLogger(__name__)
logger.info("This is an info message from a submodule.")



if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()

    logger.info("This is an info message from test script.")
    run()
    logger.info("This is an info message from test script after run() function.")

"""

# logger_setup.py
from datetime import datetime
import logging
import logging.config
import pathlib
import sys

# Import the shared global config instance
from loadConfig import config


def initialiseLogging():
    """Initializes the hierarchical logger using settings from the global config instance."""
    try:
        # Convert your custom DictToClass object to a standard Python dict
        logDictionary = dict(config.loggingConfigs)
        
        # Check your custom config's boolean toggle property
        if config.generalSettings.logging.logToFile:
            # Generate a unique timestamped file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logDirectory = pathlib.Path("logs")
            logDirectory.mkdir(exist_ok=True)
            
            # Inject the target string into the handler configuration
            logDictionary["handlers"]["file"]["filename"] = str(logDirectory / f"run_{timestamp}.log")
        else:
            # Dynamically decouple the file handler if toggled off
            logDictionary["loggers"][""]["handlers"].remove("file")
            logDictionary["handlers"].pop("file", None)
        
        # Apply the final evaluated dictionary to the framework
        logging.config.dictConfig(logDictionary)
        
    except AttributeError as error:
        # Structured fallback to avoid application crashes if keys are missing
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        logging.warning("Failed to parse logger variables from config structure: %s", error)
