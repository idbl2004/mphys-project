import os, csv, sys, logging, xspec

def setupChildLogger()-> None:
    """Sets up a logger for the child process."""
    # read the logging configuration from the environment variable
    environmentLogLevel: str = os.environ.get("LOGGING_LEVEL", "INFO").upper()

    # convert the string level to a logging level, fallback to INFO if invalid
    try:
        if hasattr(logging, "getLevelNamesMapping"):
            levelMapping: dict = logging.getLevelNamesMapping()
        else:
            levelMapping: dict = logging._nameToLevel

        numericLevel: int = levelMapping.get(environmentLogLevel, logging.INFO)
    except Exception:
        numericLevel = logging.INFO

    # configure the logger
    logging.basicConfig(
        level=numericLevel,
        stream=sys.stdout,
        format="%(levelname)s:%(message)s"
    )

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setupChildLogger()
    startRow = int(os.environ["START_ROW"])
    endRow = int(os.environ["END_ROW"])
    targetDir = os.environ["TARGET_DIR"]
    outputCsv = os.path.abspath(os.environ["OUTPUT_CSV"])

    phaFile = os.path.join(targetDir, "outputSpectrum.pha")
    rspFile = os.path.join(targetDir, "outputResponse.rsp")

    xspec.Plot.device = "none"
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.Fit.query = "yes"

    if not os.path.exists(phaFile) or not os.path.exists(rspFile):
        logger.critical("Files missing relative to root.")
        sys.exit(1)

    localRows = []

    for currentRow in range(startRow, endRow + 1):
        try:
            xspec.AllData.clear()
            xspec.AllModels.clear()
        except Exception:
            pass

        try:
            type2PhaPath = f"{phaFile}{{{currentRow}}}"
            currentSpec = xspec.Spectrum(dataFile=type2PhaPath, respFile=rspFile)

            # --- DEFENSIVE TYPE CHECKING FOR PYXSPEC DATA TRACKS ---
            rawRate = currentSpec.rate
            rawExposure = currentSpec.exposure

            # Isolate rate value if it's trapped in an array container
            if isinstance(rawRate, (tuple, list)):
                sourceRate = float(rawRate[0])
            else:
                sourceRate = float(rawRate)

            # Isolate exposure value if it's trapped in a source/background tuple container
            if isinstance(rawExposure, (tuple, list)):
                sourceExposure = float(rawExposure[0])
            else:
                sourceExposure = float(rawExposure)
            # ------------------------------------------------------

            # Safety threshold gate to safely ignore empty pre-burst slices
            if sourceRate <= 0.01 or sourceExposure <= 0.0:
                raise ValueError(f"Insufficient counts or exposure: Rate={sourceRate:.2f}")

            currentSpec.ignore("0.0-15.0 150.0-**")
            xspec.AllModels.systematic = 0.02

            bandModel = xspec.Model("grbm")
            bandModel(1).values = -1.0
            bandModel(2).values = -2.3
            bandModel(2).frozen = True
            bandModel(3).values = 50.0

            xspec.Fit.nIterations = 100
            xspec.Fit.perform()
            xspec.Fit.error("1 3")

            reducedChiSq = xspec.Fit.statistic / xspec.Fit.dof if xspec.Fit.dof > 0 else 0.0
            
            # Safely pull values out of the parameter lists
            alphaVal = bandModel(1).values[0] if isinstance(bandModel(1).values, list) else bandModel(1).values
            alphaLow, alphaHigh, _ = bandModel(1).error
            
            temVal = bandModel(3).values[0] if isinstance(bandModel(3).values, list) else bandModel(3).values
            temLow, temHigh, _ = bandModel(3).error

            # Conversion math: Epeak = E0 * (2 + alpha)
            epeakVal = temVal * (2.0 + alphaVal)
            epeakLow = temLow * (2.0 + alphaLow)
            epeakHigh = temHigh * (2.0 + alphaHigh)
            statusText = "Success"

        except Exception as sliceError:
            reducedChiSq, epeakVal, epeakLow, epeakHigh = 0.0, 0.0, 0.0, 0.0
            statusText = f"Failed: {str(sliceError)}"

        localRows.append({
            "sliceId": currentRow, 
            "status": statusText, 
            "reducedChiSq": round(reducedChiSq, 2),
            "epeak": round(epeakVal, 2), 
            "epeakLow": round(epeakLow, 2), 
            "epeakHigh": round(epeakHigh, 2)
        })

    if localRows:
        with open(outputCsv, "w", newline="") as csvFileObject:
            dictWriter = csv.DictWriter(csvFileObject, fieldnames=localRows[0].keys())
            dictWriter.writeheader()
            dictWriter.writerows(localRows)
            
    logger.info(f"BATCH_COMPLETE:{len(localRows)}")