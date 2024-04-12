from pathlib import Path
# import logging
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


sep_sc_box = [-20, -10, 360-90, 360-80]  # Klein & Hartmann 1993

def southeast_pacific_diagnostics(adfobj):
    # BOILERPLATE ADF STUFF
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output paths for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    #CAM simulation variables (this is always assumed to be a list):
    case_names = adfobj.get_cam_info("cam_case_name", required=True)

    #Grab case years
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):
        #Set obs call for observation details for plot titles
        obs = True

        #Extract variable-obs dictionary:
        var_obs_dict = adfobj.var_obs_dict

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("\t No observations found to plot against, so no zonal-mean maps will be generated.")
            return
    else:
        obs = False
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = model_rgrid_loc #Just use the re-gridded model data path
    #End if

    #Grab baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

    #Grab all case nickname(s)
    test_nicknames = adfobj.case_nicknames["test_nicknames"]
    base_nickname = adfobj.case_nicknames["base_nickname"]

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
    #-----------------------------------------

    #Set data path variables:
    #-----------------------
    mclimo_rg_loc = Path(model_rgrid_loc)
    if not adfobj.compare_obs:
        dclimo_loc  = Path(data_loc)
    #-----------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}
    # ---------------------

    seasonal_cycle_plot(adfobj, var_list, plot_locations, plot_type, case_names)
    transect_plots()
    region_climo_plots()

def transect_plots():
    pass

def region_climo_plots():
    pass

def seasonal_cycle_plot(adfobj, var_list, plot_locations, plot_type, case_names):
    ncases = len(case_names)
    print(f"[SEPAC](seasonal_cycle_plot) N_cases = {ncases}")
    if ncases > 1:
        print("STOP! -- just wanted to do one case")
        print(case_names)
        return None
    seasonal_cycle_var_list = ["SST", "CLDTOT", "SWCF"]
    res = adfobj.variable_defaults
    if adfobj.get_basic_info("compare_obs"):
        var_obs_dict = adfobj.var_obs_dict


    for v in seasonal_cycle_var_list:
        if v not in var_list:
            print(f"STOP! {v} not in var_list.")
            continue
        if adfobj.get_basic_info("compare_obs"):
            have_obs = v in var_obs_dict
            print(f"LOOK IN var_obs_dict for variable: {have_obs}")

        # else
        if v in res:
            vres = res[v]
        else:
            vres = {}
        plot_loc = Path(plot_locations[0])

        plot_name = plot_loc / f"{v}_sepac_seasonalcycle.{plot_type}"
        print(f"[SEPAC](seasonal_cycle_plot) {plot_name}")


        # load model time series files:
        m_ts_fils = get_timeseries_file(adfobj, case_names[0], v)
        print(f"FILES: {m_ts_fils}")
        mdata = load_da(m_ts_fils, v, res=vres)
        # Reduce to region:
        mdata = mdata.sel(lat=slice(sep_sc_box[0], sep_sc_box[1]), lon=slice(sep_sc_box[2], sep_sc_box[3]))
        # create annual cycle
        anncyc = mdata.groupby('time.month').mean(dim='time') # (month, lat, lon)
        avganncyc = anncyc.weighted(np.cos(np.radians(anncyc.lat))).mean(dim=("lat","lon"))
        fig, ax = plt.subplots()
        if 'lev' in mdata.dims:
            print("LEV DATA -> contour")
            mmonth, mlev = np.meshgrid(np.arange(1,13), mdata.lev)
            cn = ax.contourf(mmonth, mlev, avganncyc)
            fig.colorbar(cn)
        else:
            ax.plot(np.arange(1,13), avganncyc)
        fig.savefig(plot_name, bbox_inches='tight')


def get_timeseries_file(adfobj, case_name, field, hfile=None):
    ts_locs = adfobj.get_cam_info("cam_ts_loc", required=True)
    print(f"ts_locs is supposed to be: {ts_locs}")
    if hfile is None:
        hfile = 'h0'
        print("Defaulting to search for `*.h0.*` files")
    if len(ts_locs) > 1:
        raise NotImplemented("Multi-case not supported yet")
    ts_loc = Path(ts_locs[0])
    ts_filenames = f'{case_name}.*.{hfile}.{field}.*nc'
    ts_files = sorted(ts_loc.glob(ts_filenames))
    return ts_files


def load_dataset(fils):
    if (len(fils) == 0):
        print("Input file list is empty.")
        return None
    elif (len(fils) > 1):
        ds = xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        ds = xr.open_dataset(sfil)
    if ds is None:
        print(f"invalid data on load_dataset")
    return ds

def load_da(fils, variablename, res=None):
    da = (load_dataset(fils)[variablename]).squeeze()
    if res is None:
        res = dict()
    if variablename in res:
        vres = res[variablename]
        da = da * vres.get("scale_factor",1) + vres.get("add_offset", 0)
        da.attrs['units'] = vres.get("new_unit", da.attrs.get('units', 'none'))
    return da