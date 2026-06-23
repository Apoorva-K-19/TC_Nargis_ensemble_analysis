# TC_Nargis_ensemble_analysis

Analysis code for the study of ensemble predictability and error growth of Tropical Cyclone Nargis (2008) in machine-learning and numerical weather prediction systems. 
Compares 50-member ensembles from ECMWF IFS, GenCast, Pangu-Weather, and FourCastNetv2 across the storm lifecycle.

This repository contains the analysis and figure-generation scripts for the manuscript "Predictability and Error Growth of Tropical Cyclone Nargis (2008) in Machine Learning and Numerical Weather Prediction Systems."

## Scripts

All scripts share configurations (paths, model list, tracking parameters) defined in `track.py`; the other scripts import from it, so `track.py` must be present and its paths edited first.

| Script | Produces | Description |
|---|---|---|
| `track.py` | Figure 1 (tracks) | Two-stage MSLP-minimum tracker; ensemble track positions and ensemble-mean tracks. Also holds the shared CONFIG and ensemble loaders imported by the other scripts. |
| `intensity.py` | Figure 2 (MSLP) | Ensemble-mean MSLP time series with IQR and full-range envelopes against JTWC best-track. |
| `raincloud.py` | Figure 3 (bias/track error) | Raincloud distributions of MSLP bias and track-position error at three lifecycle phases. |
| `reliability.py` | Figure 4 (spread-error) | MSLP and track spread-to-error ratios versus forecast lead time. |
| `dte.py` | Figure 5 + Table 2 (DTE) | Difference Total Energy at 850/500/200 hPa; exponential-fit doubling times (tau2). |
| `dke.py` | Figure 6 + Table 2 (spectra) | 500 hPa kinetic-energy and difference-kinetic-energy spectra; mesoscale slope (beta). |

## Environment

The code was run under conda environment `tc_nargis`. To recreate it:

```
conda env create -f environment.yml
conda activate tc_nargis
```

Key dependencies: numpy, scipy, matplotlib, cartopy, pygrib (eccodes).
pygrib and cartopy are version-sensitive; use the pinned environment.yml.

## Input data (not included)

The raw forecast and reanalysis data are licensed and are NOT redistributed here. They must be obtained from the original sources and the paths at the top of `track.py` (the CONFIG['dirs'] block) edited to point to local copies:

- ECMWF IFS ensemble (TIGGE): ECMWF Data Store, https://ecds.ecmwf.int
- ERA5 reanalysis (initialization, pressure levels): C3S CDS, https://doi.org/10.24381/cds.bd0915c6
- GenCast / Pangu-Weather / FourCastNetv2 forecasts: produced with ECMWF's ai-models framework (https://github.com/ecmwf-lab/ai-models) via the ai-models-gencast, ai-models-panguweather, and ai-models-fourcastnetv2 plugins.
- JTWC best-track (verification): https://www.metoc.navy.mil/jtwc/jtwc.html

The scripts expect GRIB files in the per-model directories and a JTWC b-deck text file, as set in CONFIG['dirs'].

## Running

After editing paths and activating the environment, each figure is produced by running its script directly, e.g.:

```
python track.py
python dte.py
```

Each script prints diagnostic values (doubling times, slopes, biases) to stdout and saves its figure to the output path set in the script.

## Citation

If you use this code, please cite the associated manuscript (in review) and this repository via its archived Zenodo DOI.

## License

See LICENSE.
