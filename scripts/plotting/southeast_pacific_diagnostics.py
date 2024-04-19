"""Module docstring"""
from pathlib import Path
# import logging
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from adf_dataset import AdfData
import plotting_functions as pf

from collections import namedtuple

try:
    import palettable
except ImportError:
    print(f"No palettable was imported!!")

try:
    import colorcet as cc
except ImportError:
    print(f"No colorcet was imported")


## register some colormaps:
for cc_cmap in ['CET_L19', 'bmy', 'CET_D11', 'CET_D7', 'CET_D10', 'CET_L17', 'gouldian', 'CET_R3', 'CET_D1', 'bmw']:
    my_cmap = cc.cm[cc_cmap]
    mpl.colormaps.register(cmap=my_cmap, name=cc_cmap)



# Define region and transect here
Region = namedtuple("Region", "south north west east")
Transect = namedtuple("Transect", "lat west east")
sep_sc_box = Region(-20, -10, 360-90, 360-80)  # Klein & Hartmann 1993
sep_sc_transect = Transect(-20, 360-90, 360-70) # e.g., Bretherton et al. 2010 (10.5194/acpd-10-15921-2010)

#Set seasonal ranges:
seasons = {"ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]}

seasonal_cycle_var_list = ["PRECT", "SST", "CLDTOT", "SWCF", 'FSDS', 'LWCF', 'FLDS', 'FSNS', 'FSNSC', 'FLNS', 'FLNSC', 'TGCLDLWP', 'ACTREL',
                           'CDNUMC','LHFLX', 'SHFLX', 'OMEGA500', 'PBLH', 'PRECL', 'PRECC', 'PRECT', 'TMQ']

transect_var_list = ['CLOUD', 'CLDLIQ', 'T', 'Q']


def southeast_pacific_diagnostics(adfobj):
    # Start by instantiating the AdfData object
    # and Extract needed quantities from ADF object:
    adfdata = AdfData(adfobj)
    var_list = adfdata.var_list

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
            if adfdata.get_climo_file(case_name, var):
                if var in res:
                    vres = res[var]
                else:
                    vres = {}
                vres["case_label"] = adfdata.test_nicknames[case_idx]
                if var in seasonal_cycle_var_list:
                    seasonal_cycle_plot(adfdata, var, vres, plot_loc, plot_type, case_name)
                if var in transect_var_list:
                    transect_plots(adfdata, var, vres, plot_loc, plot_type, case_name)
                region_climo_plots(adfdata, var, vres, plot_loc, plot_type, case_name)
            else:
                print(f"INFO: {var = } ... will continue to next.")
                continue

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


def transect_plots(adf, var, vres, plot_loc, plot_type, case_name):
    """Plot 20°S transect of given variable."""
    have_ref = var in adf.ref_var_nam

    if have_ref:
        mdata = adf.load_regrid_da(case_name, var)
    else:    
        mdata = adf.load_climo_da(case_name, var)
    mdata = mdata.sel(lat=sep_sc_transect.lat, method='nearest').sel(lon=slice(sep_sc_transect.west, sep_sc_transect.east))

    res = adf.adf.variable_defaults
    vres = res[var] if var in res else {}

    if not have_ref:
        print(f"No reference data found for variable: {var}")
        cp_info = pf.prep_contour_plot(mdata, mdata, mdata, **vres)
    else:
        odata = adf.load_reference_da(var)
        if odata['lon'].min() < 0:
            print("Flip longitude on reference data.")
            odata = lonFlip(odata)
        odata = odata.sel(lat=sep_sc_transect.lat, method='nearest').sel(lon=slice(sep_sc_transect.west, sep_sc_transect.east))
        cp_info = pf.prep_contour_plot(mdata, odata, mdata, **vres) # can't do difference b/c we didn't ensure same size
    cmap = cp_info['cmap1']
    cnrm = cp_info['norm1']
    lvls = cp_info['levels1']
    # Check if we have PMID to move to pressure levels:
    have_pmid = "PMID" in adf.var_list
    if have_pmid:
        print(f"PMID is found, will move to pressure levels.")
        pmid = adf.load_climo_da(case_name, "PMID")
        pmid = pmid.sel(lat=sep_sc_transect.lat, method='nearest').sel(lon=slice(sep_sc_transect.west, sep_sc_transect.east))
        mdata = pf.pmid_to_plev(mdata, pmid, new_levels=100.*np.arange(100, 1020, 10), convert_to_mb=False)

    # Seasonal transects
    for season, vals in seasons.items():
        plot_name = plot_loc / f"{case_name}_{var}_{season}_sepac_20sTransect.{plot_type}"
        print(f"[SEPAC](transect_plot) {plot_name}")
        smdata = mdata.sel(time=vals).mean(dim='time')
        if 'lev' in smdata.dims:
            fig, ax = plt.subplots(ncols=2, constrained_layout=True)
            fig.suptitle(f"20°S {var} {season}")
            mlon, mlev = np.meshgrid(smdata.lon, smdata.lev)
            cnModel = ax[0].pcolormesh(mlon, mlev, smdata, norm=cnrm, cmap=cmap)
            ax[0].set_title(vres["case_label"])
            ax[0].set_ylim([1000,100])
            fig.colorbar(cnModel, ax=ax[0])
            if have_ref:
                sodata = odata.sel(time=vals).mean(dim='time')
                rlon, rlev = np.meshgrid(sodata.lon, sodata.lev)
                cnRef = ax[1].pcolormesh(rlon, rlev, sodata, norm=cnrm, cmap=cmap)
                ax[1].set_title(adf.ref_labels[var])
                ax[1].set_ylim([1000,100])
                fig.colorbar(cnRef, ax=ax[1])
        else:
            fig, ax = plt.subplots()
            ax.plot(smdata.lon, smdata, label=vres["case_label"])
            if have_ref:
                sodata = odata.sel(time=vals).mean(dim='time')
                ax.plot(sodata.lon, sodata, label=adf.ref_labels[var])
            ax.legend()
        fig.savefig(plot_name, bbox_inches='tight')
        plt.close(fig)





def region_climo_plots(adf, var, vres, plot_loc, plot_type, case_name):
    """Plot planar view of region."""
    have_ref = var in adf.ref_var_nam
    if not have_ref:
        print(f"No reference data found for variable: {var}")
        mdata = adf.load_climo_da(case_name, var)
    else:
        mdata = adf.load_regrid_da(case_name, var)
        odata = adf.load_reference_da(var)
        print(f"DEBUG: coords: {mdata.coords = }, {odata.coords = }")
        if odata['lon'].min() < 0:
            print("Flip longitude on reference data.")
            odata = lonFlip(odata)
        odata = odata.sel(lat=slice(sep_sc_box.south-10, sep_sc_box.north+10), lon=slice(sep_sc_box.west-10, sep_sc_box.east+10))
        if 'month' in odata.dims:
            odata = odata.rename({'month':'time'})
            print(f"Convert MONTH to TIME, but check it: {odata.time}")
            if (odata.time.min() == 0) and (odata.time.max() == 11):
                print(f"Time goes from 0 to 11 instead of 1 to 12 - Adjust")
                odata = odata.assign_coords({"time":odata.time+1})

    units_label = mdata.attrs.get("units", " ")
    if mdata is None:
        print(f"FAIL: climo file missing for {case_name}, {var = }")
        return None
    if mdata['lon'].min() < 0:
        print("Flip longitude on case data.")
        mdata = lonFlip(mdata)

    mdata = mdata.sel(lat=slice(sep_sc_box.south-10, sep_sc_box.north+10), lon=slice(sep_sc_box.west-10, sep_sc_box.east+10))

    # seasonal plots
    for season, vals in seasons.items():
        plot_name = plot_loc / f"{case_name}_{var}_{season}_sepac_latlon.{plot_type}"
        print(f"[SEPAC](latlon_plot) {plot_name}")
        smdata = mdata.sel(time=vals).mean(dim='time')
        if 'lev' in smdata.dims:
            print(f"Sorry, didn't implement a level selection for the map plots [{var}]")
            continue
        else:
            # this doesn't get sensible defaults for differences yet.
            if have_ref:
                sodata = odata.sel(time=vals).mean(dim='time')
                cp_info = pf.prep_contour_plot(smdata, sodata, smdata, **vres)
            else:
                cp_info = pf.prep_contour_plot(smdata, smdata, smdata, **vres)

            cmap = cp_info['cmap1']
            cnrm = cp_info['norm1']
            lvls = cp_info['levels1']
            dmap = cp_info['cmapdiff']
            dnrm = cp_info['normdiff']
            dlvl = cp_info['levelsdiff']

            fig, ax = plt.subplots(ncols=3, subplot_kw={"projection":ccrs.PlateCarree()})
            fig.suptitle(f"Southeast Pacific, {var}, {season}")
            mlon, mlat = np.meshgrid(smdata.lon, smdata.lat)
            img0 = ax[0].pcolormesh(mlon, mlat, smdata,  transform=ccrs.PlateCarree(),
                                    norm=cnrm, cmap=cmap)
            ax[0].set_title(vres["case_label"])
            CS0 = ax[0].contour(mlon, mlat, smdata, transform=ccrs.PlateCarree(), levels=lvls, colors='black')
            fig.colorbar(img0, ax=ax[0], shrink=0.3, label=units_label)
            ax[0].clabel(CS0, CS0.levels, inline=True, fontsize=10)
            # draw the region boundary
            ax[0].plot([sep_sc_box.west, sep_sc_box.east, sep_sc_box.east, sep_sc_box.west, sep_sc_box.west],
                       [sep_sc_box.south, sep_sc_box.south, sep_sc_box.north, sep_sc_box.north, sep_sc_box.south],
                       transform=ccrs.PlateCarree(), color='lightgray')
            ax[0].coastlines()
            if have_ref:
                # sodata = odata.sel(time=vals).mean(dim='time')
                rlon, rlat = np.meshgrid(sodata['lon'], sodata['lat'])
                img1 = ax[1].pcolormesh(rlon, rlat, sodata, transform=ccrs.PlateCarree(),
                                        norm=cnrm, cmap=cmap)
                CS1 = ax[1].contour(rlon, rlat, sodata, transform=ccrs.PlateCarree(), levels=lvls, colors='black')
                ax[1].clabel(CS1, CS1.levels, inline=True, fontsize=10)
                ax[1].set_title(adf.ref_labels[var])
                ax[1].plot(
                    [sep_sc_box.west, sep_sc_box.east, sep_sc_box.east, sep_sc_box.west, sep_sc_box.west],
                    [sep_sc_box.south, sep_sc_box.south, sep_sc_box.north, sep_sc_box.north, sep_sc_box.south],
                    transform=ccrs.PlateCarree(), color='lightgray')
                ax[1].coastlines()
                fig.colorbar(img1, ax=ax[1], shrink=0.3)
                # Try to plot the difference -- but only if the arrays are same size:
                if smdata.shape == sodata.shape:
                    difference = smdata-sodata
                    dimg = ax[2].pcolormesh(rlon, rlat, difference, transform=ccrs.PlateCarree(),norm=dnrm,cmap=dmap)
                    ax[2].coastlines()
                    fig.colorbar(dimg, ax=ax[2], shrink=0.3)
            else:
                ax[1].set_axis_off()
                ax[2].set_axis_off()
        fig.savefig(plot_name, bbox_inches='tight')
        plt.close(fig)

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
        odata = odata.sel(lat=slice(sep_sc_box.south, sep_sc_box.north), lon=slice(sep_sc_box.west, sep_sc_box.east))
        oanncyc = odata.weighted(np.cos(np.radians(odata.lat))).mean(dim=("lat","lon"))
        print(oanncyc)

    plot_name = plot_loc / f"{case_name}_{var}_sepac_seasonalcycle.{plot_type}"
    print(f"[SEPAC](seasonal_cycle_plot) {plot_name}")

    # load model time series files:
    # mdata = adf.load_regrid_da(case_name, v)
    mdata = adf.load_climo_da(case_name, var)

    # Reduce to region:
    mdata = mdata.sel(lat=slice(sep_sc_box.south, sep_sc_box.north), lon=slice(sep_sc_box.west, sep_sc_box.east))

    avganncyc = mdata.weighted(np.cos(np.radians(mdata.lat))).mean(dim=("lat","lon"))

    if 'lev' in mdata.dims:
        print("LEV DATA -> contour")
        fig, ax = plt.subplots(ncols=2,nrows=2,constrained_layout=True)
        mmonth, mlev = np.meshgrid(np.arange(1,13), mdata.lev)
        cnModel = ax[0,0].contourf(mmonth, mlev, avganncyc)
        ax[0,0].set_title(vres["case_label"])
        rmonth, rlev = np.meshgrid(np.arange(1,13), odata.lev)
        cnRef = ax[0,1].contourf(rmonth, rlev, oanncyc)
        ax[0,1].set_title(f"Reference")
        if avganncyc.shape == oanncyc.shape:
            print("Shapes agree, will make difference panel")
            cnDiff = ax[1,0].contourf(rmonth, rlev, avganncyc-oanncyc)
            fig.colorbar(cnDiff, ax=ax[1,0])
            ax[1,0].title(f'{vres["case_label"]} - {adf.ref_labels[var]}', loc='left')
        fig.colorbar(cnModel, ax=ax[0,0])
        fig.colorbar(cnRef, ax=ax[0,1])

    else:
        fig, ax = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
        ax[0].plot(np.arange(1,13), avganncyc, label=vres["case_label"])
        if have_ref:
            ax[0].plot(np.arange(1,13), oanncyc, label=adf.ref_labels[var], color='gray')
        ax[0].legend()
        ax[0].set_title(f"{var} SEPac annual cycle", loc='left')
        # ax[0].set_xlabel("MONTH")
        if 'units' in mdata.attrs:
            ax[0].set_ylabel(mdata.attrs['units'])

        [a.spines['top'].set_visible(False) for a in ax]
        [a.spines['right'].set_visible(False) for a in ax]
        [a.set_xlim([1,12]) for a in ax]

        if have_ref:
            ax[1].plot(np.arange(1,13), avganncyc-oanncyc)
            ax[1].set_title(f'{vres["case_label"]} - {adf.ref_labels[var]}')
            if 'units' in mdata.attrs:
                ax[1].set_ylabel(mdata.attrs['units'])
            ax[1].set_xlabel("MONTH")
        else:
            ax[1].set_axis_off()
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)