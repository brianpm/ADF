# CAM diagnostics

This repository contains the CAM diagnostics python package, which includes numerous different averaging,
re-gridding, and plotting scripts, most of which are provided by users of CAM itself.

Specifically, this package is designed to generate standard climatological comparisons between either two
different CAM simulations, or between a CAM simulation and observational and reanalysis datasets.  Ideally
this will allow for a quick evaluation of a CAM simulation, without requiring the user to generate numerous
different figures on there own.

Currently, this figure only uses standard CAM monthly (h0) outputs.  However, if there is user interest then
additional diagnostic options can be added.

## Required software environment

These diagnostics currently require Python version 3.4 or highter.  They also require the following non-standard python libraries/modules:

- PyYAML
- Numpy
- Xarray
- Matplotlib
- Cartopy

On NCAR's CISL machines (cheyenne and casper), these can be loaded by running `ncar_pylib` on the command line.  

Along with these python requirements, the `ncrcat` NetCDF Operator (NCO) is also needed.  On the CISL machines, this can be loaded by simply running `module load nco` on the command line.  

## Running CAM diagnostics

To run an example of the CAM diagnostics, simply download this repo, setup your computing environment as described in the "Required software environment" section above, modify the `config_example.yaml` file (or create one of your own) to point to the relevant diretories and run:

`./run_diag --config_file config_example.yaml`

This should generate a collection of time series files, climatology (climo) files, re-gridded climo files, and example CAM diagnostic figures, all in their respective directories.

## Troubleshooting

Any problems or issues with this script should be posted on the
DiscussCESM CAM forum located online [here](https://xenforo.cgd.ucar.edu/cesm/forums/cam.133/).

Please note that registration may be required before a message can
be posted.  However, feel free to search the forums for similar issues
(and possible solutions) without needing to register or sign in.

Good luck, and have a great day!

