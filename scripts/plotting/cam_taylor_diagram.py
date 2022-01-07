'''
Module: cam_taylor_diagram

Provides a Taylor diagram following the AMWG package. Uses spatial information only.

This module, for better or worse, provides both the computation and plotting functionality. 
It depends on an ADF instance to obtain the `climo` files. 
It is designed to have one "reference" case (could be observations) and arbitrary test cases. 
When multiple test cases are provided, they are plotted with different colors. 

'''

#
# --- imports and configuration ---
#
from pathlib import Path
import numpy as np
import xarray as xr

# import plotting_functions as pf  # No need for pf unless we move general taylor diagram into there.
import matplotlib as mpl
import matplotlib.pyplot as plt


import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning


#
# --- Main Function Shares Name with Module: cam_taylor_diagram ---
#
def cam_taylor_diagram(adfobj):
    # Extract needed quantities from ADF object:
    # -----------------------------------------
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    # Special ADF variable which contains the output path for
    # all generated plots and tables:
    plot_location = adfobj.plot_location

    case_name = adfobj.get_cam_info("cam_case_name", required=True)

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):
        data_name = "obs"  # does not get used, is just here as a placemarker
        data_list = adfobj.read_config_var("obs_type_list")  # Double caution!
        data_loc = adfobj.get_basic_info("obs_climo_loc", required=True)

    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc = adfobj.get_baseline_info("cam_climo_loc", required=True)

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"NOTE: Plot type is set to {plot_type}")

    dclimo_loc    = Path(data_loc)
    mclimo_rg_loc = Path(model_rgrid_loc)
    plot_loc      = Path(plot_location)
    #-----------------------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Check if plot output directory exists, and if not, then create it:
    if not plot_loc.is_dir():
        print("    {} not found, making new directory".format(plot_loc))
        plot_loc.mkdir(parents=True)


    # define the variables that need to be included
    # Derived: TropicalLandPrecip, TropicalOceanPrecip, EquatorialPacificStress
    # Semi-derived: U300
    #               ColumnRelativeHumidity and ColumnTemperature are WEIGHTED VERTICAL AVERAGE (All Levels)
    var_list = ['SLP', 'SWCF', 'LWCF', 
                'TropicalLandPrecip', 'TropicalOceanPrecip', 'EquatorialPacificStress', 
                'U300', 'ColumnRelativeHumidity', 'ColumnTemperature']

    # NOTE: 
    # Strategy here is to go through the list of variables and cases and get the statistics 
    # necessary for the plot. Build up the data for the plot incrementally, which should
    # provide reasonable performance (especially reduced memory pressure) when there are
    # many cases to deal with. 
    
    # generate the statistics


    # send statistics to plot

    # TODO: change plot interface to take a combined object.
    ft, axt = taylor_diagram(list(correlation.values()), np.array(list(a_sigma.values()))/np.array(list(b_sigma.values())), bias=np.array(list(bias.values())), labels=list(bias.keys()), color='red')

    pass


#
# --- Local Functions ---
#

# --- DERIVED VARIABLES --- 
def vertical_average(fld, ps):
    # get pressure field
    # get del_pressure
    # return (1/g) * sum(fld * del_pressure) / sum(del_pressure)
    pass

def get_tropical_land_precip():
    # get landfrac
    # get prect 
    # mask to only keep land locations
    pass

def get_tropical_ocean_precip():
    # get landfrac
    # get prect 
    # mask to only keep ocean locations
    pass

def get_u_at_plev():
    # parse input to know variable and pressure value
    # get variable
    # TODO: are variables saved on pressure levels, or do we need to interpolate?
    # return variable on plev
    pass

def get_virh():
    # get RELHUM climo
    # get PS
    # TODO: on pressure levels already?
    # TODO: RELHUM will have hyam, hybm? 
    # vertical_average(RELHUM, PS)
    pass

def get_vit():
    # get T climo
    # get PS
    # TODO: on pressure levels already?
    # TODO: T will have hyam, hybm? 
    # vertical_average(T, PS)
    pass


def get_derive_func(fld):
    funcs = {'TropicalLandPrecip': get_tropical_land_precip,
    'TropicalOceanPrecip': get_tropical_ocean_precip,
    'U300': get_u_at_plev,
    'VIRelativeHumidity': get_virh,
    'VITemperature': get_vit,
    }

def weighted_correlation(x, y, weights):
    # TODO: since we expect masked fields (land/ocean), need to allow for missing values (maybe works already?)
    mean_x = x.weighted(weights).mean()
    mean_y = y.weighted(weights).mean()
    dev_x = x - mean_x
    dev_y = y - mean_y
    dev_xy = dev_x * dev_y
    cov_xy = dev_xy.weighted(weights).mean()
    cov_xx = (dev_x * dev_x).weighted(weights).mean()
    cov_yy = (dev_y * dev_y).weighted(weights).mean()
    return cov_xy / np.sqrt(cov_xx * cov_yy)


def taylor_stats_single(casedata, refdata, w=True):
    """This replicates the basic functionality of 'taylor_stats' from NCL.

    input:
        casedata : input data, DataArray
        refdata  : reference case data, DataArray
        w        : if true use cos(latitude) as spatial weight, if false assume uniform weight
    
    returns: 
        pattern_correlation, ratio of standard deviation (case/ref), bias
    """
    lat = casedata['lat']
    if w: 
        wgt = np.cos(np.radians(lat))
    else:
        wgt = np.ones(len(lat))
    correlation = weighted_correlation(casedata, refdata, wgt).item()
    a_sigma = casedata.weighted(wgt).std().item()
    b_sigma = refdata.weighted(wgt).std().item()
    mean_case = casedata.weighted(wgt).mean()
    mean_ref = refdata.weighted(wgt).mean()
    bias = (100*((mean_case - mean_ref)/mean_ref)).item()
    return correlation, a_sigma/b_sigma, bias



def taylor_stats(casedata, refdata):
    """Given Dataset objects (or dictionaries with {varname:data, ...}), calculate Taylor statistics for each variable.
    
    return: datasets for correlation, std ratio, and bias
    """
    correlation = {}
    ratio = {}
    bias = {}
    for v in a_ann:
        vcorr, vratio, vbias = taylor_stats_single(casedata[v], refdata[v])
        correlation[v] = vcorr
        ratio[v] = vratio
        bias[v] = vbias
    return xr.Dataset(correlation), xr.Dataset(ratio), xr.Dataset(bias)


def taylor_diagram(correlation_ds: xr.Dataset, nstd_ds: xr.Dataset, bias=None, labels=None, ax=None, color=None):
    """Make a basic Taylor Diagram:
       correlation: 
           correlation coefficent (0-1), determines azimuthal position, xr.Dataset.
                    
       nstd: 
           normalized standard deviation, determines radial position, xr.Dataset
       
       bias: 
           if provided, same size as correlation/nstd, scales marker size, and xr.Dataset
       labels: 
           dict-like, if provided it maps from variables in correlation/nstd/bias to string labels
           If not provided, the variable names are used for labels. 
       ax: 
           the axes to plot on; this is how multiple cases can be stacked onto the same axes
       color: 
           if provided, use as the marker color
           
       Notes:
           - The datasets must contain the same variables.
           - If ax is not provided, use fig, ax = plt.subplots() to make new plot.
           - Internally, the values for each variable are put into a numpy array for plotting; the order is determined by correlation_ds.
           - When adding to an existing axes, NO PROVISION IS MADE TO MATCH ACROSS CASES. So be careful to preserver order.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'})
        # also trigger drawing
        if bias is not None:
            needs_bias_labels = True

    # originally, I just made the figure here
    #     fig, ax = plt.subplots(figsize=(8,8), subplot_kw={ 'projection':'polar'})
        
    # Stage the data into simple arrays
    # The variables in correlation_ds, nstd_ds, and (optional) bias must be the same
    assert list(correlation_ds.data_vars) == list(nstd_ds.data_vars)
    if bias is not None:
        assert list(correlation_ds.data_vars) == list(bias.data_vars)
    # if labels are provided, they need to include all the same variables:
    if labels is not None:
        for ll in labels:
            if ll not in list(correlation_ds.data_vars):
                raise IOError(f"Sorry, you provided labels, but did not include one for variable {ll}")
    final_labels = []
    correlation = []
    nstd = []
    if bias is not None:
        xbias = []
    for fld in correlation_ds:
        correlation.append(correlation_ds[fld])
        nstd.append(nstd_ds[fld])
        if (bias is not None) and (fld in bias):
            xbias.append(bias[fld])
    correlation = np.array(correlation)
    nstd = np.array(nstd)
    if bias is not None:
        xbias = np.array(xbias)

    corr_labels = np.array([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.])
    corr_locations = np.pi/2 - np.arccos((corr_labels))  # azim. ticks in radians.
    theta = np.pi/2 - np.arccos(correlation)  # Transform DATA
    
    if bias is not None:
        bias_bin = np.digitize(xbias, [-20, -10, -5, -1, 1, 5, 10, 20])
        marker_list = ["v", "v", "v", "v", "o", "^", "^", "^", "^"]
        marker_size = [24, 16, 8, 4, 4, 4, 8, 16, 24]
    
    spts = []
    for i in range(len(theta)):
        if bias is None:
            marker = 'o'
            markersize = 16
        else:
            mksz = marker_size[bias_bin[i]]
            mk = marker_list[bias_bin[i]]
            spts.append(ax.plot(theta[i], nstd[i], marker=mk, markersize=mksz, color=color))
#     spts = ax.scatter(theta, nstd, s=bias**2, marker='o', edgecolor='black', alpha=0.5)
#     print(spts)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim([0, 1.6])  # Works better than set_rmin / set_rmax
    ax.set_theta_zero_location("N") # zero at top, 
    ax.set_theta_direction(-1)  # angle increases clockwise
    thetalines, thetalabels = ax.set_thetagrids(np.degrees(corr_locations), corr_labels)
    ax.grid(axis='x', linewidth=0)  # turn off radial grid
    ax.set_rgrids(np.arange(0, 1.75, .25))
    # ax.set_yticks([0.0, 1.0, 1.25]) # same effect as set_rgrids()
    ax.set_ylabel("Standardized Deviations")
    # Add tick marks along azimuth
    tick = [ax.get_rmax(),ax.get_rmax()*0.97]
    for t in corr_locations:
        ax.plot([t,t], tick, lw=0.72, color="k")

    ax.text(np.radians(50), ax.get_rmax()*1.1, "Correlation", ha='center', rotation=-50, fontsize=15)
    ax.text(np.radians(95), 1.0, "REF", ha='center')
    
    # Annotate each point with a number and make a legend:
    if labels is not None:
        annos = []
        for i, lbl in enumerate(labels):
            annos.append(f"{i} - {lbl.replace('_','')}")
            ax.annotate(i, (theta[i], nstd[i]), ha='center', va='bottom', 
                               xytext=(0,5), textcoords='offset points', fontsize='x-large')
    textstr = "\n".join(annos)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11, bbox=props)
    
    if needs_bias_labels:
        # produce an info-box showing the markers/sizes based on bias
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerTuple
        bias_legend_elements = [(Line2D([0], [0], marker="v", color='k', label="> 20%", markersize=24, fillstyle='none', linewidth=0), Line2D([0], [0], marker="^", color='k', label="> 20%", markersize=24, fillstyle='none', linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label="10-20%", markersize=16, linewidth=0), Line2D([0], [0], marker="^", color='k', label="10-20%", markersize=16, linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label="5-10%", markersize=8, linewidth=0), Line2D([0], [0], marker="^", color='k', label="5-10%", markersize=8, linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label=">1-5%", markersize=4, linewidth=0), Line2D([0], [0], marker="^", color='k', label=">1-5%", markersize=4, linewidth=0)),
                                Line2D([0], [0], marker="o", color='k', label="< 1%", markersize=4, linewidth=0),
                               ]
        bias_legend_labels = ["> 20%", "10-20%", "5-10%", "1-5%", "< 1%"]
        ax.legend(handles=bias_legend_elements, labels=bias_legend_labels, loc='upper left', handler_map={tuple: HandlerTuple(ndivide=None, pad=2.)}, labelspacing=2, handletextpad=2, frameon=False, title=" - / + Bias",
                 title_fontsize=18)
    return ax