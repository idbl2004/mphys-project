# Installation Guide
1. Set up a virtual environment using `conda` or `pyenv`
2. Download NASA's High Energy Astrophysics software (HEASoft) package [here](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/).
>### Note
>It is **your** responsibility to check that both this package and the HEASoft installer are safe to use!
3. Clone this repo to the folder that you want to install it in.
```
git clone https://github.com/Dezray961/MVTTools
```
4. Navigate to the directory and run
```
pip install -r requirements.txt
```

Create the configured directory structure before running data tools:
```
python setupDirectories.py
```
Use `python setupDirectories.py --dry-run` to inspect the resolved paths first, or
`--config path/to/config.yaml` to use a different configuration file. Relative paths
in the configuration are resolved relative to that configuration file.

# conda environs
With the inclusion of `xspec` this now needs two seperate python environments. xspec requires python 3.10 and the rest needs 3.13. Currently I have this set up as:
* (XSpecEnv) installed with `conda create -n XspecEnv python=3.10 -c conda-forge -y` and `conda install -y -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ -c conda-forge xspec xspec-data numpy astropy scipy`

Published under the MIT License.