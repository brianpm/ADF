"""Module docstring"""
from pathlib import Path
# import logging
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from adf_dataset import AdfData



sep_sc_box = [-20, -10, 360-90, 360-80]  # Klein & Hartmann 1993
sep_sc_transect = [360-90, 360-70] # e.g., Bretherton et al. 2010 (10.5194/acpd-10-15921-2010)

#Set seasonal ranges:
seasons = {"ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]}

seasonal_cycle_var_list = ["PRECT", "SST", "CLDTOT", "SWCF", 'FSDS', 'LWCF', 'FLDS', 'FSNS', 'FSNSC', 'FLNS', 'FLNSC', 'TGCLDLWP']

transect_var_list = ['CLOUD', 'CLDLIQ', 'T', 'Q']


def southeast_pacific_diagnostics(adfobj):
    # Start by instantiating the AdfData object
    # and Extract needed quantities from ADF object:
    adfdata = AdfData(adfobj)
    var_list = adfobj.diag_var_list

    #Special ADF variable which contains the output paths for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    #Grab case years
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    #Grab baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

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

    for case_idx, case_name in enumerate(adfdata.case_names):
        #Set output plot location:
        plot_loc = Path(plot_locations[case_idx])

        #Check if plot output directory exists, and if not, then create it:
        if not plot_loc.is_dir():
            print(f"    {plot_loc} not found, making new directory")
            plot_loc.mkdir(parents=True)

        for var in var_list:
            if var in res:
                vres = res[var]
            else:
                vres = {}
            if var in seasonal_cycle_var_list:
                seasonal_cycle_plot(adfdata, var, vres, plot_loc, plot_type, case_name)
            if var in transect_var_list:
                transect_plots(adfdata, var, plot_loc, plot_type, case_name)
            region_climo_plots()


def lonFlip(data, lonname=None):
    # NOTE: this assumes global values
    if lonname is None:
        lonname = 'lon'
    tmplon = data[lonname]
    tmpdata = data.roll( {lonname: (len(tmplon) // 2)}, roll_coords=True)
    lonroll = tmpdata[lonname].values
    if tmplon.min() >= 0:
        # flip to -180:180
        tmpdata[lonname] = xr.DataArray(np.where(lonroll >= 180, lonroll - 360, lonroll), dims=[lonname], attrs=data[lonname].attrs)
    else:
        # flip from -180:180 to 0:360
        tmpdata[lonname] = xr.DataArray(((lonroll + 360) % 360), dims=[lonname], attrs=data[lonname].attrs)
    return tmpdata


def transect_plots(adf, var, plot_loc, plot_type, case_name):
    """Plot 20°S transect of given variable."""
    have_ref = var in adf.ref_var_nam
    if not have_ref:
        print(f"No reference data found for variable: {var}")
    else:
        odata = adf.load_reference_da(var)
        if odata['lon'].min() < 0:
            print("Flip longitude on reference data.")
            odata = lonFlip(odata)
        odata = odata.sel(lat=-20,method='nearest').sel(lon=slice(sep_sc_transect[0], sep_sc_transect[1]))
    
    mdata = adf.load_climo_da(case_name, var)
    mdata = mdata.sel(lat=-20,method='nearest').sel(lon=slice(sep_sc_transect[0], sep_sc_transect[1]))

    # Seasonal transects
    for season, vals in seasons.items():
        plot_name = plot_loc / f"{case_name}_{var}_{season}_sepac_20sTransect.{plot_type}"
        print(f"[SEPAC](transect_plot) {plot_name}")
        smdata = mdata.sel(time=vals).mean(dim='time')
        if 'lev' in smdata.dims:
            fig, ax = plt.subplots(ncols=2, constrained_layout=True)
            fig.suptitle(f"20°S {var} {season}")
            mlon, mlev = np.meshgrid(smdata.lon, smdata.lev)
            cnModel = ax[0].pcolormesh(mlon, mlev, smdata)
            ax[0].set_title(f"CAM")
            ax[0].set_ylim([1000,100])
            fig.colorbar(cnModel, ax=ax[0])
            if have_ref:
                sodata = odata.sel(time=vals).mean(dim='time')
                rlon, rlev = np.meshgrid(sodata.lon, sodata.lev)
                cnRef = ax[1].pcolormesh(rlon, rlev, sodata)
                ax[1].set_title(adf.ref_labels[var])
                ax[1].set_ylim([1000,100])
                fig.colorbar(cnRef, ax=ax[1])
        else:
            fig, ax = plt.subplots()
            ax.plot(smdata.lon, smdata, label="CAM")
            if have_ref:
                sodata = odata.sel(time=vals).mean(dim='time')
                ax.plot(sodata.lon, sodata, label=adf.ref_labels[var])
            ax.legend()
        fig.savefig(plot_name, bbox_inches='tight')





def region_climo_plots():
    """Plot planar view of region."""
    pass

def seasonal_cycle_plot(adf, var, vres, plot_loc, plot_type, case_name):
    """Plot the seasonal cycle.
    
    adf : adf data object
    var : str, the name of variable to plot
    vres : dict, the variable preferences
    plot_location : path to directory
    plot_type : str, extension for plots
    case_name : str, name of case to plot
    
    """
    if var not in adf.ref_var_nam:
        print(f"No reference found for variable `{var}`.")

    reference_data = adf.load_reference_dataset(var)
    if reference_data is None:
        print(f"No reference data found for variable: {var}")
        have_ref = False
    else:
        have_ref = True
        odata = adf.load_reference_da(var)
        if odata['lon'].min() < 0:
            print("Flip longitude on reference data.")
            odata = lonFlip(odata)
        odata = odata.sel(lat=slice(sep_sc_box[0], sep_sc_box[1]), lon=slice(sep_sc_box[2], sep_sc_box[3]))
        oanncyc = odata.weighted(np.cos(np.radians(odata.lat))).mean(dim=("lat","lon"))
        print(oanncyc)

    plot_name = plot_loc / f"{case_name}_{var}_sepac_seasonalcycle.{plot_type}"
    print(f"[SEPAC](seasonal_cycle_plot) {plot_name}")

    # load model time series files:
    # mdata = adf.load_regrid_da(case_name, v)
    mdata = adf.load_climo_da(case_name, var)

    # Reduce to region:
    mdata = mdata.sel(lat=slice(sep_sc_box[0], sep_sc_box[1]), lon=slice(sep_sc_box[2], sep_sc_box[3]))

    avganncyc = mdata.weighted(np.cos(np.radians(mdata.lat))).mean(dim=("lat","lon"))

    if 'lev' in mdata.dims:
        print("LEV DATA -> contour")
        fig, ax = plt.subplots(ncols=2,nrows=2,constrained_layout=True)
        mmonth, mlev = np.meshgrid(np.arange(1,13), mdata.lev)
        cnModel = ax[0,0].contourf(mmonth, mlev, avganncyc)
        ax[0,0].set_title(f"CAM")
        rmonth, rlev = np.meshgrid(np.arange(1,13), odata.lev)
        cnRef = ax[0,1].contourf(rmonth, rlev, oanncyc)
        ax[0,1].set_title(f"Reference")
        if avganncyc.shape == oanncyc.shape:
            print("Shapes agree, will make difference panel")
            cnDiff = ax[1,0].contourf(rmonth, rlev, avganncyc-oanncyc)
            fig.colorbar(cnDiff, ax=ax[1,0])
            ax[1,0].title("Model - Ref", loc='left')
        fig.colorbar(cnModel, ax=ax[0,0])
        fig.colorbar(cnRef, ax=ax[0,1])

    else:
        fig, ax = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
        ax[0].plot(np.arange(1,13), avganncyc, label='CAM')
        ax[0].plot(np.arange(1,13), oanncyc, label=adf.ref_labels[var], color='gray')
        ax[0].legend()
        ax[0].set_title(f"{var} SEPac annual cycle", loc='left')
        # ax[0].set_xlabel("MONTH")
        if 'units' in mdata.attrs:
            ax[0].set_ylabel(mdata.attrs['units'])
        print(f"COORDS: {avganncyc.coords = }, {oanncyc.coords = }")
        ax[1].plot(np.arange(1,13), avganncyc-oanncyc)
        ax[1].set_title("CAM - Reference")
        if 'units' in mdata.attrs:
            ax[1].set_ylabel(mdata.attrs['units'])
        ax[1].set_xlabel("MONTH")
        [a.spines['top'].set_visible(False) for a in ax]
        [a.spines['right'].set_visible(False) for a in ax]
        [a.set_xlim([1,12]) for a in ax]


    fig.savefig(plot_name, bbox_inches='tight')