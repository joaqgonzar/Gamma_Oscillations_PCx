# Gamma_Oscillations_PCx

This repository contains all python codes to reproduce Figures 1-7 of
Gonzalez, J., Torterolo, P., & Tort, A. B. L. (2023). Mechanisms and functions of respiration-driven gamma oscillations in the primary olfactory cortex. eLife, 12, e83044. https://doi.org/10.7554/eLife.83044

The datasets should be downloaded from: (https://crcns.org/data-sets/pcx/pcx-1). The preprocessed LFPs can be downloaded from (http://gofile.me/5vylI/LLisQ1sUd). Alternatively, the raw files should be decimated and downsampled to 2000 Hz in order to perform all analyses.

All codes start from the raw data, run all analysis, and plot the figures from the paper.

To start, the extension to all files in the processed folders should be changed to .mat. Specifically, xxx_bank1.efd files should be renamed to xxx_bank_efd.mat, the same with bank2, while the xxx.resp files to xxx_resp.mat   


