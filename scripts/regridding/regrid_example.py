def regrid_example(case_name, input_climo_loc, output_loc,
                   var_list, target_list, target_loc,
                   overwrite_regrid):

    """
    This is an example function showing how to set-up a
    re-gridding method for CAM climatology (climo) files,
    so that they are on the same grid as observations or
    baseline climatologies.

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name"
    input_climo_loc  -> Location of CAM climo files provided by "cam_climo_loc"
    output_loc       -> Location to write re-gridded CAM files, specified by "cam_regrid_loc"
    var_list         -> List of CAM output variables provided by "diag_var_list"
    target_list      -> List of target data sets CAM could be regridded to
    taget_loc        -> Location of target files that CAM will be regridded to
    overwrite_regrid -> Logical to determine if already existing re-gridded
                        files will be overwritten. Specified by "cam_overwrite_regrid"
    """

    #Import necessary modules:
    import xarray as xr
    import numpy as np

    from pathlib import Path

    # regridding
    # Try just using the xarray method
    # import xesmf as xe  # This package is for regridding, and is just one potential solution.

    # Steps:
    # - load climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - regrid one to the other (probably should be a choice)

    #Notify user that script has started:
    print("  Regridding CAM climatologies...")

    #Set input/output data path variables:
    #------------------------------------
    mclimo_loc  = Path(input_climo_loc)
    rgclimo_loc = Path(output_loc)
    tclimo_loc  = Path(target_loc)
    #------------------------------------

    #Check if re-gridded directory exists, and if not, then create it:
    if not rgclimo_loc.is_dir():
        print("    {} not found, making new directory".format(rgclimo_loc))
        rgclimo_loc.mkdir(parents=True)

    # probably want to do this one variable at a time:
    for var in var_list:
        print("\t [\u25B6] regridding {} (known targets: {})".format(var, len(target_list)))
        #loop over regridding targets:
        for target in target_list:

            #Determine regridded variable file name:
            regridded_file_loc = rgclimo_loc / '{}_{}_{}_regridded.nc'.format(target, case_name, var)

            #Check if re-gridded file already exists and over-writing is allowed:
            if regridded_file_loc.is_file() and overwrite_regrid:
                #If so, then delete current file:
                regridded_file_loc.unlink()

            #Check again if re-gridded file already exists:
            if not regridded_file_loc.is_file():

                #Create list of regridding target files (we should explore intake as an alternative to having this kind of repeated code)
                tclim_fils = sorted(list(tclimo_loc.glob("{}*_{}_*.nc".format(target, var))))

                if len(tclim_fils) > 1:
                    #Combine all target files together into a single data set:
                    tclim_ds = xr.open_mfdataset(tclim_fils, combine='by_coords')
                else:
                    #Open single file as new xarray dataset:
                    tclim_ds = xr.open_dataset(tclim_fils[0])

                #Generate CAM climatology (climo) file list:
                mclim_fils = sorted(list(mclimo_loc.glob("{}_{}_*.nc".format(case_name, var))))

                if len(mclim_fils) > 1:
                    #Combine all cam files together into a single data set:
                    mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
                else:
                    #Open single file as new xsarray dataset:
                    mclim_ds = xr.open_dataset(mclim_fils[0])

                #Extract variable info from model data:
                mdata = mclim_ds[var]

                #Extract grid info from target data:
                if 'time' in tclim_ds.coords:
                    tgrid = tclim_ds.isel(time=0).squeeze()
                else:
                    tgrid = tclim_ds

                #Regrid model data to match target grid:
                rgdata = regrid_data(mdata, tgrid, method=1)

                # Collect any of the "special" variables:
                rgdata = rgdata.to_dataset()
                for special in ["hyam", "hybm", "hyai", "hybi", "P0"]:
                    if special in mclim_ds:
                        rgdata[special] = mclim_ds[special]
                # also check for PS in mdata... regrid it when found:
                if "PS" in mclim_ds:
                    rgdata['PS'] = regrid_data(mclim_ds['PS'], tgrid, method=1)

                #Write re-gridded data to output file:
                save_to_nc(rgdata, regridded_file_loc)

    #Notify user that script has ended:
    print("[\u2705]...CAM climatologies have been regridded successfully.")

#################
#Helper functions
#################

def save_to_nc(tosave, outname, attrs=None, proc=None):
    """Saves xarray variable to new netCDF file"""

    #Import necessary modules:
    import xarray as xr

    xo = tosave  # used to have more stuff here.
    # deal with getting non-nan fill values.
    if isinstance(xo, xr.Dataset):
        enc_dv = {xname: {'_FillValue': None} for xname in xo.data_vars}
    else:
        enc_dv = dict()
    enc_c = {xname: {'_FillValue': None} for xname in xo.coords}
    enc = {**enc_c, **enc_dv}
    if attrs is not None:
        xo.attrs = attrs
    if proc is not None:
        xo.attrs['Processing_info'] = f"Start from file {origname}. " + proc
    xo.to_netcdf(outname, format='NETCDF4', encoding=enc)

def regrid_data(fromthis, tothis, method=1):
    """Regrid data using various different methods"""

    #Import necessary modules:
    import xarray as xr

    if method == 1:
        # kludgy: spatial regridding only, seems like can't automatically deal with time
        if 'time' in fromthis.coords:
            result = [fromthis.isel(time=t).interp_like(tothis) for t,time in enumerate(fromthis['time'])]
            result = xr.concat(result, 'time')
            return result
        else:
            return fromthis.interp_like(tothis)
    elif method == 2:
        newlat = tothis['lat']
        newlon = tothis['lon']
        coords = dict(fromthis.coords)
        coords['lat'] = newlat
        coords['lon'] = newlon
        return fromthis.interp(coords)
    elif method == 3:
        newlat = tothis['lat']
        newlon = tothis['lon']
        ds_out = xr.Dataset({'lat': newlat, 'lon': newlon})
        regridder = xe.Regridder(fromthis, ds_out, 'bilinear')
        return regridder(fromthis)
    elif method==4:
        # geocat
        newlat = tothis['lat']
        newlon = tothis['lon']
        result = geocat.comp.linint2(fromthis, newlon, newlat, False)
        result.name = fromthis.name
        return result

