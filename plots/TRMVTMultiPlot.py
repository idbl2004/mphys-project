"""
Plots the time-resolved MVT for different window sizes using photon counts from a CSV file. 
"""


from pathlib import Path
import importlib

from ruamel.yaml import YAML


# setup yaml stuff
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.preserve_quotes = True

projectRootPath = Path(__file__).resolve().parent
configFilePath = "config.yaml"

# load the config file
with open(configFilePath, "r") as f:
    config = yaml.load(f)

# set the slice duration for the time-resolved MVT calculation
sliceDuration = 60.0 # seconds

# update the config file with the new slice duration
config['preProcessingConfig']['swiftBATConfig']['processing']['sliceDuration'] = sliceDuration

# save the updated config file
with open(configFilePath, "w") as f:
    yaml.dump(config, f)

from time import time
import numpy as np
import loadConfig
import analysisTools.haarMethods as haarMethods
import swiftDataTools.swiftBATPipe as swiftBATPipe

importlib.reload(loadConfig)
importlib.reload(haarMethods)
importlib.reload(swiftBATPipe)

from loadConfig import config

processSwiftBATData = swiftBATPipe.processSwiftBATData
haarPowerMod = haarMethods.haarPowerMod
timeResolvedMVT = haarMethods.timeResolvedMVT
haarDenoise = haarMethods.haarDenoise

#startTime = time()
#
#grbName: str = "GRB080319B"
#processSwiftBATData(
#    GRBName = grbName,
#    download = False,
#    deleteOriginal = False,
#    )
#
#endTime = time()
#print(f"Total processing time: {endTime - startTime:.2f} seconds")

def csvReader(
        filepath: str
    ) -> tuple:
    """
    Reads a CSV file containing photon counts and errors, returning them as numpy arrays.
    """
    filePath: str = filepath + "/photonCounts.csv"
    data: np.ndarray = np.genfromtxt(filePath, delimiter=',', skip_header=1)
    counts: np.ndarray = data[:, 0]
    errors: np.ndarray = data[:, 1]
    return counts, errors


# get the photon counts and errors from the CSV file
photonCounts, photonErrors = csvReader("/home/derekpinkett/coding/MVTTools/data/processed/GRB080319B")

# denoise the photon counts
#photonCounts = haarDenoise(photonCounts, photonErrors)




import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import numpy as np

windowSizes = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


fig, axs = plt.subplots(
    len(windowSizes)//2,
    len(windowSizes)//(len(windowSizes)//2),
    figsize=(10, 6),
    sharex=True
)
axs = axs.flatten()

for i, windowSize in enumerate(windowSizes):
    results = timeResolvedMVT(
        photonCounts,
        photonErrors,
        source = "swift",
        timeWindowSizeSeconds = windowSize
    )

    times = []
    MVTs = []
    MVTErrors = []

    for result in results:
        times.append(result['centerTimeSeconds'])
        MVTs.append(result['mvtMs'])
        MVTErrors.append(result['mvtErrMs'])

    # convert the lists to numpy arrays for easier plotting
    times = np.array(times)
    MVTs = np.array(MVTs)
    MVTErrors = np.array(MVTErrors)

    # 1. FIX: Convert zeros, negatives, and infinities into NaNs to create gaps
    invalid_indices = ~np.isfinite(MVTs) | (MVTs <= 0)
    MVTs[invalid_indices] = np.nan

    ax = axs[i]
    
    # 2. Plot the array (Matplotlib will automatically leave gaps for the NaNs)
    ax.plot(
        times,
        MVTs,
        label='MVT',
        color='blue',
        linewidth=1.0
    )

    at = offsetbox.AnchoredText(
        f'Window Size: {windowSize} s', 
        loc='upper right',           
        frameon=True,                
        prop=dict(size=10)
    )

    at.patch.set_boxstyle("round,pad=0.2")
    at.patch.set_alpha(0.8)          
    ax.add_artist(at)
    
    # FIX: Remove the ax.set_xticklabels([]) manual override
    if i >= 4: 
        ax.set_xlabel('Time [s]')
        
    if i % 2 != 0: 
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    
    ax.set_ylabel('MVT [$m$s]')


    ax.axvline(
        x=2,
        color='grey',
        linestyle='--',
        label='Trigger Time'
    )

    ax.hlines(
        y=40,
        xmin=np.nanmin(times),
        xmax=np.nanmax(times),
        color='grey',
        linestyle='-',
        label='GB14 result'
    )

    # 3. FIX: Calculate limits safely ignoring the NaNs and factoring in the y=40 line
    if i == 2:
        ax.set_ylim(0, 600)
    elif not np.all(np.isnan(MVTs)):
        # nanmax ignores the NaN gaps so your limit stays true to the real data
        max_data_val = np.nanmax(MVTs)
        max_val = max(max_data_val, 40) 
        ax.set_ylim(0, max_val * 1.1)
    else:
        ax.set_ylim(0, 600)
    print(f"Window Size: {windowSize} s, complete.")


plt.subplots_adjust(wspace=0, hspace=0)
axs[-1].legend(loc='upper left')
plt.show()
