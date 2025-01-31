from pathlib import Path
import numpy as np
import xarray as xr

import xesmf

import plotting_functions as pf

import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning

#Set seasonal ranges:
seasons = {"ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]}


use_regrid_files = False # use climo files if False


# HARD CODE HERE -- TO BE MOVED INTO ADF OBJECT:
regrid_weights_file = Path("/glade/work/brianpm/mapping_ne30pg3_to_fv09_esmfbilin.nc")

plot_name_string = "COSP_CALIPSO_MAP"

def cosp_calipso_2501(adfobj):

    ######
    print(f"VERSIONS: xarray: {xr.__version__}, xesmf: {xesmf.__version__}, numpy: {np.__version__}")
    ######

    # we have specific variables to get here instead of full list from adfobj
    var_list = ["CLDTOT_CAL"]

    # Standard ADF stuff:
    plot_locations = adfobj.plot_location
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]
    res = adfobj.variable_defaults
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    #
    # SECTION 1: determine which plots need to be made
    #
    for case_idx, case_name in enumerate(adfobj.data.case_names):
        #Set output plot location:
        plot_loc = Path(plot_locations[case_idx])

        #Loop over the variables for each season
        skip_make_plot = []
        for var in var_list:
            for s in seasons:
                plot_name = plot_loc / f"{var}_{s}_{plot_name_string}.{plot_type}"
                print(f"Projected file name: {plot_name}. Exists: {plot_name.is_file()}")
                # normally we now check whether to re-do plot or skip
                # for now we always do the plot, so skip_make_plot is empty list
    #
    # SECTION 2: get the variable-specific options
    #
    for var in var_list:
        if var not in adfobj.data.ref_var_nam:
            dmsg = f"No reference data found for variable `{var}`."
            adfobj.debug_log(dmsg)
            print(dmsg)
            has_ref = False             # this might mean skip entirely
        else:
            has_ref = True

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"cosp_calipso_2501: Found variable defaults for {var}")
        else:
            vres = {}

        # load reference data (observational or baseline)
        if not adfobj.compare_obs:
            base_name = adfobj.data.ref_case_label
        else:
            base_name = adfobj.data.ref_labels[var]

        # 
        # KEY ISSUE HERE: until we have better regridding - load the climo files and regrid
        #
        if use_regrid_files:
            odata = adfobj.data.load_reference_regrid_da(base_name, var)
            if odata is None:
                dmsg = f"No regridded baseline file for {base_name} for variable `{var}`, zonal mean plotting skipped."
                adfobj.debug_log(dmsg)
                continue
        else:
            odata = adfobj.data.load_reference_climo_da(case_name, var)
            print(odata)

        # check if odata needs regridding:
        has_lat_lon = pf.lat_lon_validate_dims(odata)
        if not has_lat_lon:
            print("Variable looks like it needs regridding")
            print(f"Has ncol: {'ncol' in odata.dims}")
            # Method kwarg MUST MATCH how the weights file was generated
            R = make_se_regridder(regrid_weights_file, Method='bilinear')
            # overwrite since we want to proceed with regridded
            odata = regrid_se_data_bilinear(R, odata)

        # process cases
        for case_idx, case_name in enumerate(adfobj.data.case_names):
            case_nickname = adfobj.data.test_nicknames[case_idx]
            plot_loc = Path(plot_locations[case_idx])
            if use_regrid_files:
                mdata = adfobj.data.load_regrid_da(case_name, var)
            else:
                mdata = adfobj.data.load_climo_da(case_name, var)
            if mdata is None:
                dmsg = f"No test file for {case_name} for variable `{var}`, plotting skipped."
                adfobj.debug_log(dmsg)
                continue
            has_lat_lon = pf.lat_lon_validate_dims(mdata)
            if not has_lat_lon:
                print(f"Variable looks like it needs regridding. Has ncol: {'ncol' in mdata.dims}")
                R = make_se_regridder(regrid_weights_file, Method='bilinear')
                # overwrite since we want to proceed with regridded
                mdata = regrid_se_data_bilinear(R, mdata)
            #
            # Seasonal Averages
            #
            mseasons = {}
            oseasons = {}
            for s in seasons:
                #Seasonal Averages
                mseasons[s] = pf.seasonal_mean(mdata, season=s, is_climo=True)
                oseasons[s] = pf.seasonal_mean(odata, season=s, is_climo=True)
                diffld = mseasons[s] - oseasons[s]
                pctdiff = 100*(diffld / oseasons[s])
                plot_name = plot_loc / f"{var}_{s}_{plot_name_string}.{plot_type}"

                if plot_name not in skip_make_plot:
                    pf.plot_map_and_save(plot_name, case_nickname, adfobj.data.ref_nickname,
                                        [syear_cases, eyear_cases], [syear_baseline, eyear_baseline],
                                        mseasons[s], oseasons[s], diffld, pctdiff, **vres)
                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, season=s, plot_type="LatLon")
                    print(f"Plot done: {plot_name}")
    #Notify user that script has ended:
    print("  ...COSP plots have been generated successfully.")

# THIS IS FROM WILL WIEDER'S PR: 
# Regrids unstructured SE grid to regular lat-lon
# Shamelessly borrowed from @maritsandstad with NorESM who deserves credit for this work
# https://github.com/NorESMhub/xesmf_clm_fates_diagnostic/blob/main/src/xesmf_clm_fates_diagnostic/plotting_methods.py


def make_se_regridder(weight_file, Method='conservative'):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )
    # Apparently not needed ... maybe for Method=conservative ?
    # # Hard code masks for now, not sure this does anything?
    # s_mask = xr.DataArray(s_data.data.reshape(in_shape[0],in_shape[1]), dims=("lat", "lon"))
    # dummy_in['mask']= s_mask
    
    # d_mask = xr.DataArray(d_data.values, dims=("lat", "lon"))  
    # dummy_out['mask']= d_mask                

    # do source and destination grids need masks here?
    # See xesmf docs https://xesmf.readthedocs.io/en/stable/notebooks/Masking.html#Regridding-with-a-mask
    regridder = xesmf.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        # results seem insensitive to this method choice
        # choices are coservative_normed, coservative, and bilinear
        method=Method,
        reuse_weights=True,
        periodic=True,
    )
    return regridder

def regrid_se_data_bilinear(regridder, data_to_regrid, column_dim_name='ncol'):
    if isinstance(data_to_regrid, xr.Dataset):
        vars_with_ncol = [name for name in data_to_regrid.variables if column_dim_name in data_to_regrid[name].dims]
        updated = data_to_regrid.copy().update(data_to_regrid[vars_with_ncol].transpose(..., "ncol").expand_dims("dummy", axis=-2))
    elif isinstance(data_to_regrid, xr.DataArray):
        updated = data_to_regrid.transpose(...,column_dim_name).expand_dims("dummy",axis=-2)
    else:
        raise ValueError(f"Something is wrong because the data to regrid isn't xarray: {type(data_to_regrid)}")
    regridded = regridder(updated)
    return regridded