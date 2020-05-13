from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf

def zonal_mean(case_name, model_rgrid_loc, data_name, data_loc,
                 var_list, data_list, plot_location):

    """
    This script plots zonal averages.
    Compare CAM climatologies against
    other climatological data (observations or baseline runs).

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
    data_loc         -> Location of comparison data, which is either "obs_climo_loc"
                        or "cam_baseline_climo_loc", depending on whether
                        "compare_obs" is true or false.
    var_list         -> List of CAM output variables provided by "diag_var_list"
    data_list        -> List of data sets CAM will be compared against, which
                        is simply the baseline case name in situations when
                        "compare_obs" is false.
    plot_location    -> Location where plot files will be written to, which is
                        specified by "cam_diag_plot_loc".
    Notes:
        The script produces plots of 2-D and 3-D variables,
        but needs to determine which type along the way. 
        For 3-D variables, the default behavior is to interpolate
        climo files to pressure levels, which requires the hybrid-sigma
        coefficients and surface pressure. That ASSUMES that the climo
        files are using native hybrid-sigma levels rather than being 
        transformed to pressure levels.
    """

    print("  Generating zonal mean plots...")

    #Set input/output data path variables:
    #------------------------------------
    dclimo_loc    = Path(data_loc)
    mclimo_rg_loc = Path(model_rgrid_loc)
    plot_root     = Path(plot_location)
    plot_loc      = plot_root / '{}_vs_{}'.format(case_name, data_name)
    #-----------------------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Set plot file type:
    plot_type = 'png'

    #Check if plot output directory exists, and if not, then create it:
    if not plot_loc.is_dir():
        print("    {} not found, making new directory".format(plot_loc))
        plot_loc.mkdir(parents=True)

    print("HERE IS THE VARLIST: {}".format(var_list))
    for var in var_list:
        print("\t \u231B zonal mean plots for {}".format(var))
        #loop over different data sets to plot model against:
        for data_src in data_list:
            # load data (observational) comparison files 
            # (we should explore intake as an alternative to having this kind of repeated code):
            oclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))
            oclim_ds = _load_dataset(oclim_fils)

            # load re-gridded model files:
            mclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))
            mclim_ds = _load_dataset(mclim_fils)

            #Extract variable of interest
            odata = oclim_ds[var].squeeze()  # squeeze in case of degenerate dimensions
            mdata = mclim_ds[var].squeeze()

            # determine whether it's 2D or 3D
            # 3D triggers search for surface pressure
            has_lat, has_lev = pf.zm_validate_dims(mdata)  # assumes will work for both mdata & odata
            if has_lev:
                print("{} has lev dimension.".format(var))
                # need hyam, hybm, PS, P0 for both datasets
                if 'hyam' not in mclim_ds:
                    print("\u2757 PROBLEM -- NO hyam")
                    print(mclim_ds)
                mhya = mclim_ds['hyam']
                mhyb = mclim_ds['hybm']
                if 'time' in mhya.dims:
                    mhya = mhya.isel(time=0).squeeze()
                if 'P0' in mclim_ds:
                    P0 = mclim_ds['P0']
                else:
                    P0 = 100000.0  # Pa
                if 'PS' in mclim_ds:
                    mps = mclim_ds['PS']
                else:
                    # look for the file (this isn't great b/c we'd have to constantly re-load)
                    mps_files = sorted(list(mclimo_rg_loc.glob("{}_{}_PS_*.nc".format(data_src, case_name))))
                    if len(mps_files) > 0:
                        mps_ds = _load_dataset(mps_files)
                        mps = mps_ds['PS']
                    else:
                        continue  # what else could we do?

                # We need a way to check whether 'obs' or 'baseline' here:
                # (and/or a way to specify pressure levels or hybrid levels)
                ohya = oclim_ds['hyam']
                ohyb = oclim_ds['hybm']
                if 'PS' in oclim_ds:
                    ops = oclim_ds['PS']
                else:
                    # look for the file (this isn't great b/c we'd have to constantly re-load)
                    ops_files = sorted(list(dclimo_loc.glob("{}_PS_*.nc".format(data_src))))
                    if len(ops_files) > 0:
                        ops_ds = _load_dataset(ops_files)
                        ops = ops_ds['PS']
                    else:
                        continue  # what else could we do?
            else:
                mhya = mhyb = ohya = ohyb = None

            #
            # Seasonal Averages
            # Note: xarray can do seasonal averaging, but depends on having time accessor, which these prototype climo files don't.
            #

            #Create new dictionaries:
            mseasons = dict()
            oseasons = dict()
            dseasons = dict() # hold the differences

            #Loop over season dictionary:
            for s in seasons:
                mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')

                if has_lev:
                    mps_season = mps.sel(time=seasons[s]).mean(dim='time')
                    ops_season = ops.sel(time=seasons[s]).mean(dim='time')
                else:
                    mps_season = ops_season = None

                # difference: each entry should be (lat, lon) or (plev, lat, lon)
                # dseasons[s] = mseasons[s] - oseasons[s]
                # difference will be calculated in plot_zonal_mean_and_save;
                # because we can let any pressure-level interpolation happen there
                # This could be re-visited for efficiency or improved code structure.

                # time to make plot; here we'd probably loop over whatever plots we want for this variable
                # I'll just call this one "Mean_LatLon"  ... would this work as a pattern [operation]_[AxesDescription] ?
                # NOTE: Up to this point, nothing really differs from plot_example, except to deal with getting PS ready,
                #       so we could have made one script instead of two.
                #       Merging would make overall timing better because looping twice will double I/O steps.
                # 
                plot_name = plot_loc / "{}_{}_Zonal_Mean.{}".format(var, s, plot_type)

                #Remove old plot, if it already exists:
                if plot_name.is_file():
                    plot_name.unlink()

                #Create new plot:
                pf.plot_zonal_mean_and_save(plot_name, mseasons[s], mps_season, mhya, mhyb, 
                                                       oseasons[s], ops_season, ohya, ohyb)

    #Notify user that script has ended:
    print("\u2705  ...Zonal mean plots have been generated successfully.")


#
# Helpers
#
def _load_dataset(fils):
    if len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)