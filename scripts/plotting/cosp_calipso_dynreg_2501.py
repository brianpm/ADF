"""
Dynamical Regimes Analysis

Use a 'dynamical variable' to define regimes (usually OMEGA500).

For each variable in `variable_list` make a plot of the conditional
average in bins of the dynamical variable. They can be 2D or 3D,
and that determines if they are plotted as lines on a single axes
or panels.

This script is targeted at COSP variable which may need to be
regridded. All regridding is based on `regrid_weights_file` at this time.

Parameter settings:
- xbins
- regrid_weights_file
- remove_land
- region definitions (plots made for each region)

Note: this will print the paths for the plots,
but they are not yet added to ADF web site generator.

"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xarray as xr

import xesmf

import plotting_functions as pf
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"
warnings.formatwarning = my_formatwarning

# Dataclass Regions is very simple, and assumes no longitudinal structure.
@dataclass
class Regions:
    name: str
    latitudes: slice
    label: str

    def region_definition(self):
        """Make a string that specified the latitude bounds."""
        val1 = self.latitudes.start
        if val1 > 0:
            hem1 = "N"
        elif val1 < 0:
            hem1 = "S"
        else:
            hem1 = ""
        val2 = self.latitudes.stop
        if val2 > 0:
            hem2 = "N"
        elif val2 < 0:
            hem2 = "S"
        else:
            hem2 = ""
        if np.absolute(val1) > np.absolute(val2):
            return f"{np.absolute(val2)}{hem2}-{np.absolute(val1)}{hem1}"
        else:
            return f"{np.absolute(val1)}{hem1}-{np.absolute(val2)}{hem2}"


# Set seasonal ranges:
seasons = {
    "ANN": np.arange(1, 13, 1),
    "DJF": [12, 1, 2],
    "JJA": [6, 7, 8],
    "MAM": [3, 4, 5],
    "SON": [9, 10, 11],
}

# PARAMETER SETTINGS
xbins = np.arange(-100, 110, 10)  # hPa/day

plot_name_string = "COSP_CALIPSO_DYNREG"

cnorm = mpl.colors.Normalize(vmin=0.0, vmax=100)

# HARD CODE HERE -- TO BE MOVED INTO ADF OBJECT:
regrid_weights_file = Path("/glade/work/brianpm/mapping_ne30pg3_to_fv09_esmfbilin.nc")

remove_land = True
regions = {"global":Regions('global', slice(-60,60), "Global Ocean"),
          "tropics":Regions('tropics', slice(-30,30), "Tropical Ocean"),
          "socean":Regions('socean', slice(-60,-30), "Southern Ocean")} 

# constructed bins include under and over:
bin_ctr = xbins[0:-1] + 0.5 * np.diff(xbins)
bin_ctr = np.insert(bin_ctr, 0, xbins[0] - 0.5 * (xbins[1] - xbins[0]))
bin_ctr = np.append(bin_ctr, xbins[-1] + 0.5 * (xbins[-1] - xbins[-2]))
#
# "Main" function
#
def cosp_calipso_dynreg_2501(adfobj):

    ######
    print(
        f"VERSIONS: xarray: {xr.__version__}, xesmf: {xesmf.__version__}, numpy: {np.__version__}"
    )
    ######

    # we have specific variables to get here instead of full list from adfobj
    # `var_list` are the variables to be studied,
    # `load_var_list` are all the variables that need to be loaded
    dyn_var = "OMEGA500"
    var_list = ["CLD_CAL", "CLDTOT_CAL"]
    print(f"Dynamical regimes analysis script. Dynamics variable: {dyn_var}, Variable list: {var_list}")
    load_var_list = var_list + [dyn_var]
    if remove_land:
        load_var_list += ["LANDFRAC"]

    # Standard ADF stuff:
    plot_locations = adfobj.plot_location
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]
    res = adfobj.variable_defaults
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get("plot_type", "png")
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info("redo_plot")
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    print("*** PRINT CASE NAMES ***")
    print(adfobj.data.case_names)

    #
    # SECTION 1: determine which plots need to be made
    #
    for case_idx, case_name in enumerate(adfobj.data.case_names):
        # Set output plot location:
        plot_loc = Path(plot_locations[case_idx])

        # Loop over the variables for each season
        skip_make_plot = []
        for var in var_list:
            for s in seasons:
                plot_name = plot_loc / f"{var}_{s}_{plot_name_string}.{plot_type}"
                print(
                    f"Projected file name: {plot_name}. Exists: {plot_name.is_file()}"
                )
                # normally we now check whether to re-do plot or skip
                # for now we always do the plot, so skip_make_plot is empty list
    #
    # SECTION 2: load the time series files, regrid, make merged dataset
    #

    case_datasets = {}

    # deal with "reference case" first:
    time_series_arrays = {}
    for var in load_var_list:
        ts_da = adfobj.data.load_reference_timeseries_da(var)
        # if unstructured, regrid:
        has_lat_lon = pf.lat_lon_validate_dims(ts_da)
        if not has_lat_lon:
            print("Variable looks like it needs regridding")
            print(f"Has ncol: {'ncol' in ts_da.dims}")
            # Method kwarg MUST MATCH how the weights file was generated
            R = make_se_regridder(regrid_weights_file, Method="bilinear")
            # overwrite since we want to proceed with regridded
            ts_da = regrid_se_data_bilinear(R, ts_da)
        time_series_arrays[var] = ts_da.copy()
        print(f"RETRIEVED TIME SERIES for {adfobj.data.ref_case_label}, {var}")
    case_datasets[adfobj.data.ref_case_label] = xr.Dataset(time_series_arrays)

    for case_idx, case_name in enumerate(adfobj.data.case_names):
        time_series_arrays = {}
        for var in load_var_list:
            ts_da = adfobj.data.load_timeseries_da(case_name, var)
            # if unstructured, regrid:
            has_lat_lon = pf.lat_lon_validate_dims(ts_da)
            if not has_lat_lon:
                print("Variable looks like it needs regridding")
                print(f"Has ncol: {'ncol' in ts_da.dims}")
                # Method kwarg MUST MATCH how the weights file was generated
                R = make_se_regridder(regrid_weights_file, Method="bilinear")
                # overwrite since we want to proceed with regridded
                ts_da = regrid_se_data_bilinear(R, ts_da)
            time_series_arrays[var] = ts_da.copy()
            print(f"RETRIEVED TIME SERIES for {case_name}, {var}")
        case_datasets[case_name] = xr.Dataset(time_series_arrays)
    
    if remove_land:
        for c, ds in case_datasets.items():
            landfrac = ds['LANDFRAC']
            for v in ds.data_vars:
                if v != "LANDFRAC":
                    ds[v] = rm_land(ds[v], landfrac)

    #
    # Section 3: make histogram of dynamical variable (e.g., OMEGA500)
    #
    whist = {}
    binned_vars = {} # datasets casename: [var_list]
    vdim = {}
    for i, ds in case_datasets.items():
        if "OMEGA500" in ds:
            print("identified omega500")
            if ds["OMEGA500"].max() < 10:
                print(
                    f"inferring that OMEGA500 is in Pa/s ---> units says: {ds['OMEGA500'].attrs.get('units', '')}"
                )
                if "hPa" not in ds["OMEGA500"].attrs.get("units",''):
                    print("double checked and `hPa` not in units, so will transform")
                    ds["OMEGA500"] *= 864.0
                    ds["OMEGA500"].attrs["units"] = "hPa d**-1"
        hbins = xbins
        hbins = np.insert(hbins, 0, np.finfo(float).min)
        hbins = np.append(hbins, np.finfo(float).max)
        dynvar_histogram, _ = np.histogram(
            ds[dyn_var].values.flatten(), bins=hbins, density=True
        )
        dynvar_histogram = dynvar_histogram * np.diff(_)  # converts to probability mass function (sum = 1.0)
        dynvar_histogram = xr.DataArray(
            dynvar_histogram,
            dims=[f"{dyn_var}_bin_center"],
            coords={f"{dyn_var}_bin_center": bin_ctr},
            name=dyn_var,
        )
        whist[i] = dynvar_histogram

        # Section 4: Conditional Average in dyn_var bins
        # NO REGIONS YET
        binned_case = []
        xvariable = dyn_var
        x = ds[xvariable]
        for yvariable in var_list:
            y = ds[yvariable]
            if y.shape == x.shape:
                y_bs = binned_mean(x, y, xbins)
                yxr = xr.DataArray(
                    y_bs,
                    dims=f"{xvariable}_bin_center",
                    coords={f"{xvariable}_bin_center": bin_ctr},
                    name=yvariable,
                )
                binned_case.append(yxr.copy())
                if yvariable not in vdim:
                    vdim[yvariable] = 0 # flag for line plot
            else:
                if len(y.shape) == (len(x.shape) + 1):
                    zdim = set(y.dims).difference(x.dims).pop()
                    zcoord = y[zdim]  # hack to get the name of the height dimension
                    y_bs = binned_mean(x, y, xbins)
                    # the bins that come back include a beginning bin for anything (-inf, xbins[xvariable][0])
                    yxr = xr.DataArray(
                        y_bs,
                        dims=[f"{xvariable}_bin_center", zdim],
                        coords={f"{xvariable}_bin_center": bin_ctr, zdim: zcoord},
                        name=yvariable,
                    )
                    binned_case.append(yxr.copy())
                    if yvariable not in vdim:
                        vdim[yvariable] = 1 # flag for panel plot
        binned_vars[c] = xr.merge(binned_case)

    print("HISTOGRAM")
    print(whist)
    print("BINNED AVERAGES")
    print(binned_vars)


    # PLOTS
    # Make a plot of the OMEGA500 histograms:
    fig, ax = plt.subplots()
    for i, ds in whist.items():
        ax.plot(ds[f'{dyn_var}_bin_center'], ds, label=i)
    ax.legend()
    plot_name = plot_loc / f"{dyn_var}_GlobalHistogram.{plot_type}"
    fig.savefig(plot_name)
    print(f"Saved {dyn_var} Histogram: {str(plot_name)}")


    for v in var_list:
        if vdim[v] == 0:
            fig2, ax2 = plt.subplots()
            for i, ds in binned_vars.items():
                ax2.plot(ds[f'{dyn_var}_bin_center'], ds[v], label=i)
            ax2.set_title(v)
            ax2.set_xlabel(dyn_var)
            ax2.legend()
            plot_name = plot_loc / f"{v}_in_{dyn_var}_bins_Global.{plot_type}"
            fig.savefig(plot_name)
            print(f"Saved {v}({dyn_var}): {str(plot_name)}")
        elif vdim[v] == 1:
            fig2, ax2 = plt.subplots(nrows=len(binned_vars))
            mb, mz = np.meshgrid(ds[f'{dyn_var}_bin_center'], ds[zcoord])
            rownum = 0
            for row, data in binned_vars.items():
                ax2[rownum].pcolormesh(mb, mz, data[v].transpose(), shading="auto", norm=cnorm)
                ax2[rownum].contour(
                    mb,
                    mz,
                    data[v].transpose(),
                    levels=100 * np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                    colors="gray",
                )
                rownum += 1
            plot_name = plot_loc / f"{v}_in_{dyn_var}_bins_Global.{plot_type}"
            fig2.savefig(plot_name)
            print(f"Saved {v}({dyn_var}): {str(plot_name)}")


# REGRIDDING BASED ON WILL WIEDER'S ADF PR:
# Regrids unstructured SE grid to regular lat-lon
# Shamelessly borrowed from @maritsandstad with NorESM who deserves credit for this work
# https://github.com/NorESMhub/xesmf_clm_fates_diagnostic/blob/main/src/xesmf_clm_fates_diagnostic/plotting_methods.py
def make_se_regridder(weight_file, Method="conservative"):
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


def regrid_se_data_bilinear(regridder, data_to_regrid, column_dim_name="ncol"):
    if isinstance(data_to_regrid, xr.Dataset):
        vars_with_ncol = [
            name
            for name in data_to_regrid.variables
            if column_dim_name in data_to_regrid[name].dims
        ]
        updated = data_to_regrid.copy().update(
            data_to_regrid[vars_with_ncol]
            .transpose(..., "ncol")
            .expand_dims("dummy", axis=-2)
        )
    elif isinstance(data_to_regrid, xr.DataArray):
        updated = data_to_regrid.transpose(..., column_dim_name).expand_dims(
            "dummy", axis=-2
        )
    else:
        raise ValueError(
            f"Something is wrong because the data to regrid isn't xarray: {type(data_to_regrid)}"
        )
    regridded = regridder(updated)
    return regridded


#
# Analysis methods
#


def get_coslat(data, latname="lat"):
    return np.cos(np.radians(data[latname]))


def binned_mean(xvar, yvar, xbins):
    """
    a "fast" binned statistic that works in the vertical based on 2d independent variable
    The use case for this was to plot `clcalipso` in bins of Omega500.

    xvar : the independent variable (2 dimensional)
    yvar : the dependent variable (can be 2 or 3 dimensional)
    xbins : the bin edges to use
    """
    # the weighted average in each bin is SUM[w(bin)*x(bin)] / SUM[w(bin)]
    # THIS PART ONLY NEEDS DOING ONCE:
    binnumber = np.digitize(
        xvar, xbins
    )  ## says which bin each (time,lat,lon) points belongs in
    # Define area-weights
    # wgt = np.cos(np.radians(xvar.lat))
    wgt = get_coslat(xvar)
    warr = wgt.broadcast_like(xvar)  ## weights the same size as xvar

    nbins = len(xbins)  # Need one more bin
    # print(f"{nbins = }")

    xrank = len(xvar.shape)
    yrank = len(yvar.shape)
    xvarnan = np.isnan(xvar)  # compute this once, and use below.
    if xrank == yrank:
        # arrays need to be only valid data in both xvar and yvar
        dataValid = ~(xvarnan | np.isnan(yvar))  # the values that we keep
        weightedData = (warr * yvar).values[dataValid]
        binAssign = binnumber[
            dataValid
        ]  ## only keep the ones where xvar&yvar are valid
        # flatcount = np.bincount(binAssign.ravel())  ## counts the values in each bin (reduces to size of bins)
        wgtValid = warr.values[dataValid]
        sumWeightsPerBin = np.bincount(
            binAssign.ravel(), weights=wgtValid.ravel(), minlength=nbins + 1
        )
        sumWeightedDataPerBin = np.bincount(
            binAssign.ravel(), weights=(weightedData).ravel(), minlength=nbins + 1
        )
        # print(f"{sumWeightsPerBin = }, {sumWeightedDataPerBin = }")
        weightedAveragePerBin = sumWeightedDataPerBin / sumWeightsPerBin
    elif xrank == (yrank - 1):
        # only allow one dimension extra in y, identify it by dim name:
        loopDimName = set(yvar.dims).difference(xvar.dims).pop()
        # <alt version>
        # yDimsInX = [d in xvar.dims for d in yvar.dims]
        # assert np.sum(yDimsInX) == 1, "Dimension names do not match"
        # loopDimName = yvar.dims[yDimsInX.index(False)]
        # </alt version>
        # initialize the result array
        weightedAveragePerBin = np.zeros((nbins + 1, len(yvar[loopDimName])))
        # print(f"{weightedAveragePerBin.shape = }")
        for k, lev in enumerate(yvar[loopDimName]):
            ylev = yvar.isel({loopDimName: k})  # this should be same shape as xvar
            dataValid = ~(xvarnan | np.isnan(ylev))  # the values that we keep
            weightedData = (warr * ylev).values[dataValid]
            binAssign = binnumber[dataValid]
            # print(f"{k = }, {binAssign.shape = }, uniq: {np.unique(binAssign)}")
            wgtValid = warr.values[dataValid]
            sumWeightsPerBin = np.bincount(
                binAssign.ravel(), weights=wgtValid.ravel(), minlength=nbins
            )
            sumWeightedDataPerBin = np.bincount(
                binAssign.ravel(), weights=(weightedData).ravel(), minlength=nbins
            )
            # print(f"{sumWeightsPerBin = }, {sumWeightedDataPerBin = }")
            weightedAveragePerBin[:, k] = sumWeightedDataPerBin / sumWeightsPerBin
    else:
        raise NotImplementedError(
            f"Sorry, dimensions are too different: {xvar.shape = }, {yvar.shape = }"
        )

    return weightedAveragePerBin

def rm_land(dset, lf):
    """Return dset with np.nan where lf <= 0. 
    If lat dimensions are same size, but not equal, will reassign
    lat from dset to lf and report largest difference.
    """
    assert hasattr(dset, 'lat'), f'No lat in data set, {dset.coords = }'
    assert len(dset['lat']) == len(lf['lat']), f"NO GOOD! data: {len(dset['lat'])}, land: {len(lf['lat'])}"
    if np.count_nonzero(dset['lat'] == lf['lat']) != len(lf['lat']):
        print(f"[rm_land] Latitudes mismatch. Largest discrepancy is {np.max(np.absolute(dset['lat'].values - lf['lat'].values))} degrees.")
        lf = lf.assign_coords({"lat":dset["lat"], "lon":dset["lon"]})
    return xr.where(lf <= 0, dset, np.nan)