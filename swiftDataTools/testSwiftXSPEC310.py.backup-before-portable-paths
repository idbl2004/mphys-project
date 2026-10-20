"""
Find the time resolved Epeak for a given set of Swift BAT data.

This script is designed to run the swiftXSPEC310.py script in parallel across multiple CPU cores,
dividing the workload into chunks of rows to be processed. Each chunk is handled by a separate Python
3.10 process, which writes its results to a temporary scratch CSV file. Once all chunks are processed,
the script consolidates the results, sorts them by sliceId, and writes the final output to a single CSV
file.

It is designed to be run in a Python 3.13 environment, but it spawns isolated Python 3.10 processes to
handle the XSPEC analysis, which is not compatible with Python 3.13.

30/07/2026 - Derek Pinkett
It is currently a standalone script and needs to be refined into a class that can be called from the main
processing pipeline. 

It currently returns results dominated by nonphysical fits. This is because the Swift BAT has an energy
resolution of 15-150 keV (max 350 keV but the mask is transparent above 150 keV) and the spectral peak
of GRB 080319B (the test data) is ~600 keV, well outside the energy range of the instrument. This means
that there is not enough information in the data to properly constrain the spectral fits, and the results
are dominated by nonphysical fits. Dave has sugggested a number of avenues to explore:
* Look at Fermi GBM data for the same burst, which has a much wider energy range and see if that can be
    used to constrain the fits. (This might require a whole project to make statistically significant
    conclusions about parameters?)
* Look into using a different spectral model. Currently it is using a band function; this requires 4
    parameters to be fit. A cutoff power law only requires 3 parameters but one of them is the Epeak,
    which is the parameter of interest. (A personal thought: could the cutoff power law be used to
    constrain the Epeak and then use that as a prior for the band function fit?)
* Look into how the value of ~600 keV was determined for the Epeak of GRB 080319B. What assumptions were
    made and could these assumptions be used to constrain the fits?


I am temporarily shelving this aspect of the project as has become a can of worms. It potentially
requires processing of Fermi GBM data, which is further down my to-do list. If there is time I will
return to it after the Fermi GBM pipeis complete. Otherwise, you will have to pick up this aspect.
"""


import os, time, csv, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()


# =====================================================================
# GLOBAL CONFIGURATION 
# =====================================================================
PYTHON_310_PATH = "/home/derekpinkett/.conda/envs/XspecEnv/bin/python"
SCRIPT_310_PATH = "/home/derekpinkett/coding/MVTTools/swiftDataTools/swiftXSPEC310.py"
TARGET_FOLDER   = "data/reproc/00306757000/bat/spectral"

OUTPUT_DIR      = "/home/derekpinkett/coding/MVTTools/results" 
OUTPUT_FILENAME = "type2ChunkedResults.csv"

START_ROW = 1
END_ROW   = 227  
# =====================================================================


def processChildLogs(
        process: subprocess.Popen,
        parentLoggerName: str
    ):
    """Reads the stdout of a child process and logs it to the parent logger."""
    logger = logging.getLogger(parentLoggerName)

    # map the string to the logging level
    logMapping = {
        "INFO": logger.INFO,
        "CRITICAL": logger.CRITICAL
    }

    # loop through the stdout of the child process
    for line in process.stdout:
        cleanLine = line.strip()
        if cleanLine:
            # extract the prefix (INFO, CRITICAL, etc.) and the message
            parts = cleanLine.split(" ", 1)
            prefix = parts[0]

            # match the level tag to its logging level
            if prefix in logMapping and len(parts) > 1:
                logMessage = parts[1]
                logMapping[prefix]("[Python 3.10] %s", logMessage)
            else:
                logger.info("[Python 3.10 RAW] %s", cleanLine)

    # Ensure the child process has completed
    process.wait()

    if process.returncode == 0:
        logger.info("Child process completed successfully.")
    else:
        logger.error("Child process exited with return code: %d", process.returncode)


def runChunkWorker(start, end, targetFolder, pythonExecutable, outputDir, chunkIndex):
    """Spawns an isolated Python process writing to a private scratch file."""
    scratchCsv = os.path.join(outputDir, f"scratch_chunk_{chunkIndex}.csv")

    # Set up the environment variables for the subprocess
    localEnv = os.environ.copy()
    localEnv["START_ROW"] = str(start)
    localEnv["END_ROW"] = str(end)
    localEnv["TARGET_DIR"] = targetFolder
    localEnv["OUTPUT_CSV"] = scratchCsv
    localEnv["PYTHONUNBUFFERED"] = "1"
    localEnv["LOGGING_LEVEL"] = "INFO"
    
    command = [pythonExecutable, SCRIPT_310_PATH]
    
    slicesCount = 0
    try:
        with subprocess.Popen(
            command,
            env=localEnv,
            stdout=subprocess.PIPE,
            text=True
            ) as process:
            for line in process.stdout:
                if "BATCH_COMPLETE" in line:
                    slicesCount = int(line.split(":")[-1])
    except Exception:
        logger.error("An error occurred while processing the child process.")
        
    return slicesCount, scratchCsv

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    finalCsvPath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if os.path.exists(finalCsvPath):
        os.remove(finalCsvPath)

    # Work distribution calculations
    maxWorkers = max(1, os.cpu_count() - 2)
    totalRows = END_ROW - START_ROW + 1
    chunkSize = max(1, totalRows // maxWorkers)

    chunks = []
    currentStart = START_ROW
    idx = 0
    while currentStart <= END_ROW:
        currentEnd = min(END_ROW, currentStart + chunkSize - 1)
        chunks.append((currentStart, currentEnd, idx))
        currentStart = currentEnd + 1
        idx += 1

    logger.info(f"Divided {totalRows} rows into {len(chunks)} parallel blocks across {maxWorkers} CPU cores...")
    
    startTime = time.time()
    scratchFiles = []
    
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        futures = [
            executor.submit(runChunkWorker, start, end, TARGET_FOLDER, PYTHON_310_PATH, OUTPUT_DIR, i)
            for start, end, i in chunks
        ]
        
        with tqdm(total=totalRows, desc="Fitting Slices", unit="slice") as progressBar:
            for future in as_completed(futures):
                slicesProcessed, scratchPath = future.result()
                progressBar.update(slicesProcessed)
                if os.path.exists(scratchPath):
                    scratchFiles.append(scratchPath)
                
    # --- COMBINE, SORT, AND CLEAN STEP ---
    logger.info("\nConsolidating and sorting worker scratch outputs...")
    aggregatedData = []
    fieldnames = None
    
    for scratchFile in scratchFiles:
        with open(scratchFile, 'r') as f:
            reader = csv.DictReader(f)
            if not fieldnames:
                fieldnames = reader.fieldnames
            for row in reader:
                # Convert sliceId to integer to allow mathematical chronological sorting
                row['sliceId'] = int(row['sliceId'])
                aggregatedData.append(row)
        os.remove(scratchFile) # Delete temporary file footprints instantly

    # Sort everything perfectly by sliceId
    aggregatedData.sort(key=lambda x: x['sliceId'])

    # Write out the clean master dataset file
    if aggregatedData:
        with open(finalCsvPath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregatedData)

    elapsedTime = time.time() - startTime
    logger.info(f"{len(aggregatedData)} rows ordered and compiled into: {finalCsvPath}")
    logger.info(f"Processing complete in {elapsedTime:.2f} seconds.")
