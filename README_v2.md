# MVTTools

Tools for measuring the minimum variability timescale (MVT) of gamma-ray bursts
using the Haar wavelet method (Golkhou & Butler 2014), following the validation
workflow of Bala et al. (2026).

This is a fork of [Dezray961/MVTTools](https://github.com/Dezray961/MVTTools) by
Derek Pinkett, used for the MPhys project *Identifying Potential "Capris"
Gamma-Ray Bursts Through Precursor Properties* (University of Bath).

# Installation Guide

These steps were tested on macOS with an Apple chip.

### 1. Install conda (Miniforge)
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Close and reopen the terminal afterwards.

### 2. Clone this repo
```
git clone https://github.com/rodrigogleon/MVTTools
cd MVTTools
```

### 3. Create the main environment (Python 3.13)
```
conda create -n MVTEnv python=3.13 -c conda-forge -y
conda activate MVTEnv
pip install -r requirements_working.txt
gdt-data init
```

### 4. Create a second environment with HEASoft
HEASoft is needed because `analysisPipe.py` imports the Swift tools, which use
`heasoftpy`. We keep `MVTEnv` as a clean backup and add HEASoft to a copy:
```
conda create -n MVTEnvHEA --clone MVTEnv
conda activate MVTEnvHEA
conda install heasoft -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ -c conda-forge
conda deactivate
conda activate MVTEnvHEA
```
`heasoftpy` comes bundled with HEASoft. **Use `MVTEnvHEA` to run the code.**

### 5. Set the data paths
In `config.yaml`, change `dataPath` and `processedDataPath` to folders on your
own machine. Then check and create the directory structure:
```
python setupDirectories.py --dry-run
python setupDirectories.py
```
Use `--config path/to/config.yaml` to use a different configuration file.
Relative paths in the configuration are resolved relative to that file.

### 6. Check everything works
```
python -c "import analysisTools.analysisPipe"
```
If this prints nothing, the installation is complete.

> ### Note
> It is **your** responsibility to check that this package and the HEASoft
> installer are safe to use!

# Known issues

* `requirements.txt` (original) lists `gdt`, which is an unrelated bioinformatics
  package. The Fermi tools are `astro-gdt` and `astro-gdt-fermi`, which are
  included in `requirements_working.txt`.
* `heasoftpy` cannot be installed with pip. It is installed with HEASoft (step 4).
* `astro-gdt` requires matplotlib below 3.11, so `requirements_working.txt` uses
  matplotlib 3.10.9.
* `.gitignore` ignores all `*.txt` files. Use `git add -f` to commit new `.txt` files.

# Xspec environment (for E_peak fitting)

Xspec needs a separate Python 3.10 environment (set up by Derek; not yet tested
in this fork):
```
conda create -n XspecEnv python=3.10 -c conda-forge -y
conda install -y -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ -c conda-forge xspec xspec-data numpy astropy scipy
```
Note: NASA has discontinued the `xspec-data` package from HEASoft 6.37 onwards
and recommends the `ftgetmodeldata` utility instead.

Published under the MIT License. Original code by Derek Pinkett.