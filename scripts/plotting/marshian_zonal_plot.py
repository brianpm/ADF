# Zonal plots with log axis.

# note: dan's plots used xarray's plotting utilities, which work great, but I changed to matplotlib
# - this requires more work, but allows more fine-grained control. 

# There are many ways to build up the plots. I like to have a bunch of small functions, but it's just a style choice.


# Libraries we expect to be available:
from pathlib import Path
import warnings
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def marshian_zonal_plot(adfobj):
    """This function ONLY get the information from ADF and then runs the functions below."""
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    case1_dict = adfobj.read_config_var("diag_cam_climo")
    case2_dict = adfobj.read_config_var("diag_cam_baseline_climo")
    case1_name = case1_dict["cam_case_name"]
    case1_loc = case1_dict["cam_ts_loc"]
    case2_name = case2_dict["cam_case_name"]
    case2_loc = case2_dict["cam_ts_loc"]
    plot_location = basic_info_dict['cam_diag_plot_loc']
    plot_log_u_t(case1_name, case1_loc, case2_name, case2_loc, plot_location)
    
    

def plot_log_u_t(case1_name, case1_loc, case2_name, case2_loc, plot_location):
    """The driver for this module. 
       This is the function that needs to interact with ADF.

    case1_name : string that gives the name of the first case
    case2_name : string that gives the name of the second case

    N.B., difference plots are case2 MINUS case1

    case1_loc, case2_loc: Path to the case files (climo files expected)

    plot_location: path to directory where plots should be saved.


    Follows the zonal_mean.py example, but this script ONLY plots T and U, so 
    we don't need a `var_list` argument (removed). Also remove `data_list` from 
    that example. Rename to just have case1 and case2. 
    
    """
    variables = ["U", "T"]  # this is our variable list

    # ASSUME WE ARE LOADING MONTHLY CLIMO.

    for v in variables:
        # USE OUR HELPER SCRIPT TO LOAD DATASETS
        case1ds = _load_dataset(case1_loc, case1_name, v, other_name=case2_name)
        case2ds = _load_dataset(case2_loc, case2_name, v, other_name=case1_name)
        ## just in case
        if case1ds is None:
            case1ds = _load_dataset_lastditch(case1_loc, case1_name)
        if case2ds is None:
            case2ds = _load_dataset_lastditch(case2_loc, case2_name)

        # EXTRACT THE DataArray
        case1data = case1ds[v].squeeze()  # squeeze in case of degenerate dimensions
        case2data = case2ds[v].squeeze()

        # NOTE: do we need to interpolate to pressure levels?
        #       At high levels, CAM/WACCM are pure pressure anyway.

        # Apply zonal average
        case1data = case1data.mean(dim='lon')
        case2data = case2data.mean(dim='lon')
        
        # get monthly climatology
        case1data = case1data.groupby("time.month").mean(dim='time')
        case2data = case2data.groupby("time.month").mean(dim='time')
        case1data = case1data.rename({"month":"time"})
        case2data = case2data.rename({"month":"time"})
        
        print(case1data)
        
        # Apply seasonal average (function included below should probably be included in ADF library)
        case1seasons = _seasonal_average(case1data)
        case2seasons = _seasonal_average(case2data)

        
        # in case:
        case1seasons = case1seasons.compute()
        case2seasons = case2seasons.compute()

        # call the plot function:
        # plots functions should just take data to be plotted, and should not do much computation
        # In this case, the only computation that is done is taking the difference.
        f, a = season_plot(case1seasons, case2seasons, v)

        # NOTE: ADF should supply a method for constructing the output file name b/c we need it to conform to the website generator expectation.
        fnam = _plot_file_name(case1_name, case2_name, v, plot_location)

        f.savefig(fnam, bbox_inches='tight', dpi=300)


def season_plot(d1, d2, v):
    """Plot DJF and JJA in colums, either U or T"""

    # set parameters based on variable:
    if v == "T":
        levels = np.arange(130,300,10)
        cmap='magma'
        norm = mpl.colors.Normalize(vmin=130, vmax=300)
        label="T (K)"
    elif v == "U":
        levels = np.arange(-120,121,20)
        cmap='PuOr_r'
        norm = mpl.colors.TwoSlopeNorm(vmin=-120, vcenter=0.0, vmax=120)
        label="U (m/s)"
    # Parameters for difference
    DIFFERENCE = d2-d1
    mxdiff = np.max([0, np.abs(DIFFERENCE).max().item()])
    dcmap = 'coolwarm'
    dnorm = mpl.colors.TwoSlopeNorm(vmin=-1*mxdiff, vcenter=0, vmax=mxdiff)

    glat, glev = np.meshgrid(d1.lat, d1.lev)
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16,15), constrained_layout=True, sharex=True, sharey=True)
    # since we know exactly what we want, hard code each ax element:
    ax[0,0].contourf(glat,glev,d1.sel(season='DJF'), levels=levels, cmap=cmap, norm=norm)
    ax[0,0].contour(glat,glev,d1.sel(season='DJF'), levels=levels, colors='black')
    ax[0,1].contourf(glat,glev, d1.sel(season='JJA'), levels=levels, cmap=cmap, norm=norm)
    ax[0,1].contour(glat,glev, d1.sel(season='JJA'), levels=levels, colors='black')

    ax[1,0].contourf(glat,glev,d1.sel(season='DJF'), levels=levels, cmap=cmap, norm=norm)
    ax[1,0].contour(glat,glev,d1.sel(season='DJF'), levels=levels, colors='black')
    ax[1,1].contourf(glat,glev, d1.sel(season='JJA'), levels=levels, cmap=cmap, norm=norm)
    ax[1,1].contour(glat,glev, d1.sel(season='JJA'), levels=levels, colors='black')

    ax[2,0].contourf(glat, glev, DIFFERENCE.sel(season='DJF'), levels=11, cmap=dcmap, norm=dnorm)
    ax[2,1].contourf(glat, glev, DIFFERENCE.sel(season='JJA'), levels=11, cmap=dcmap, norm=dnorm)
    
    [a.invert_yaxis() for a in ax.ravel()]
    [a.set_yscale("log") for a in ax.ravel()]
    [a.set_ylim([1000,.01]) for a in ax.ravel()]

    ax[0,0].set_title("DJF", fontsize=18, fontweight='bold')
    ax[0,1].set_title("JJA", fontsize=18, fontweight='bold')
    ax[-1,0].set_xlabel("LATITUDE", fontsize=18, fontweight='bold')
    ax[-1,0].set_ylabel("PRESSURE", fontsize=18, fontweight='bold')
    cax1 = plt.axes([1., 0.37, 0.03, 0.6])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax1, orientation='vertical', label=label)
    cax2 = plt.axes([1., 0.05, 0.03, 0.27])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=dnorm, cmap=dcmap), cax=cax2, orientation='vertical', label=label)
    return fig, ax


#
# Helpers
#
def _load_dataset(data_loc, case_name, variable, other_name=None):
    """
    This method exists to get an xarray Dataset that can be passed into the plotting methods.
    
    This could (should) be changed to use an intake-esm catalog if (when) that is available.
    * At some point, we hope ADF will provide functions that can be used directly to replace this step,
      so the user will not need to know how the data gets saved.

    In this example, assume climo files are available via the ADF api.

    The complexity is because the data may have been regridded to match the "other" grid,
    in which case filies are named 

    ADF creates climo files that 
    
    """

    dloc    = Path(data_loc)

    # a hack here: ADF uses different file names for "reference" case and regridded model data,
    # - try the longer name first (regridded), then try the shorter name
    if other_name is not None:
        fils = sorted(list(dloc.glob("{}_{}_{}_*.nc".format(other_name, case_name, variable))))
    else:
        # in case "other" is not known, just look for it
        fils = sorted(list(dloc.glob("*_{}_{}_*.nc".format(case_name, variable))))
    if len(fils) == 0:
        fils = sorted(list(dloc.glob("{}_{}_*.nc".format(case_name, variable))))

    if len(fils) == 0:
        warnings.warn(f"Input file list is empty.")
        return None
    elif len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)


def _seasonal_average(darr):
    """Return seasonal averages from monthly climatology. Includes annual average too.
    **NOTE**: Does not weight by length of months.
    """
    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}
    savg = dict()

    for s in seasons:
        savg[s] = darr.sel(time=seasons[s]).mean(dim='time')

    c = xr.concat(savg.values(), dim='season')
    c = c.assign_coords({"season":list(seasons.keys())})

    return c


def _plot_file_name(c1, c2, variable, plocation):
    return Path(plocation) / f"{variable}_djf_jja_zonal_log.png"



def _load_dataset_lastditch(c_loc, c_nam):
    from pathlib import Path
    dloc = Path(c_loc)
    fils = sorted(list(dloc.glob(f"{c_nam}*.nc")))
    if len(fils) == 0:
        raise IOError("Could not find dataset")
    elif len(fils) == 1:
        return xr.open_dataset(fils[0])
    else:
        return xr.open_mfdataset(fils)
    