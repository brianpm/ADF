import xarray as xr
import numpy as np
import geocat.comp as gc  # use geocat's interpolation
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path 


def california_stratocumulus(adfobj):
    """Main function for making the California stratocumulus profile plot.
    
    Makes profiles of specific humidity, potential temperature, and cloud liquid water
    in the region 20N-30N, 230E-240E. Uses the climatological mean files, makes Annual and Seasonal average plots.
    Plots each case versus baseline/reference case separately. 
    Interpolates to pressure levels 600hPa to 1000hPa by 15 hPa.
    """
    # Check that we have the required variables:
    var_list = adfobj.diag_var_list
    our_required_variables = ['CLDLIQ', 'PS', 'T', 'Q']
    for req in our_required_variables:
        assert req in var_list, f'Sorry variable: {req} is required for California stratocumulus plot'

    # ADF-specified location for plot output    
    plot_locations = adfobj.plot_location
    plot_type = adfobj.get_basic_info('plot_location', required=False)
    if plot_type is None:
        plot_type = 'png'

    #
    # Case Names and Locations
    #
    case_names = adfobj.get_cam_info('cam_case_name', required=True)  # Loop over these
    case_climo_loc = adfobj.get_cam_info('cam_climo_loc', required=True)
    data_name = adfobj.get_baseline_info('cam_case_name', required=True)
    data_loc = adfobj.get_baseline_info("cam_climo_loc", required=True)

    seasons = {"ANN": np.arange(1,13,1), "DJF": [12, 1, 2], "JJA": [6, 7, 8], "MAM": [3, 4, 5], "SON": [9, 10, 11]}
    # define domain
    latslice = slice(20,30)
    lonslice = slice(230,240) # Klein&Hartmann 1993, Table 1
    levels = 100.*np.arange(600.0, 1015., 15)  # chosen for convenience. Go finer if native grid is finer.
    ref_ds = process_case(data_loc, latslice, lonslice, levels)

    for i, c in enumerate(case_names):
        case_ds = process_case(case_climo_loc[i], latslice, lonslice, levels)
        for s in seasons:
            ref_season = ref_ds.sel(time=seasons[s]).mean(dim='time')
            case_season = case_ds.sel(time=seasons[s]).mean(dim='time')
            # ** Ready to make plot **
            labels = [data_name, c]
            casefig, caseax = make_plot(ref_season, case_season, labels)
            casefig.suptitle(f"California Stratocumulus, {s}")

            if isinstance(plot_locations, list):
                assert len(plot_locations) == len(case_names), 'Plot locations is a list, so should provide locations per case.'
                plot_loc = plot_locations[i]
            else:
                plot_loc = plot_locations
            plot_name = Path(plot_loc) / f"TQL_{s}_CalSc_Mean.{plot_type}"
            #Remove old plot, if it already exists:
            if plot_name.is_file():
                plot_name.unlink()
            casefig.savefig(plot_name, bbox_inches='tight')
            print(f"California Stratocumulus Plot: completed {s}. File: {plot_name}")       
#
#  ---- local functions ----
#
def make_plot(ds1, ds2, caselabels):
    fig, ax = plt.subplots(figsize=(9,4), ncols=3, sharey=True, constrained_layout=True)
    custom_cycler = (cycler(color=['k', 'r', 'y', 'y']) +
                 cycler(lw=[2, 2, 1, 1]))
    [a.set_prop_cycle(custom_cycler) for a in ax]
    if ds1['Q'].max() < 0.1:
        qscale = 1000. # kg/kg -> mg/kg
    else:
        qscale = 1
    if ds1['CLDLIQ'].max() < 0.001:
        lscale = 1000.*1000.  # kg/kg -> ug/kg
    else:
        lscale = 1
    ax[0].plot(ds1['Q']*qscale, ds1['plev']/100, label=caselabels[0])
    ax[0].plot(ds2['Q']*qscale, ds2['plev']/100, label=caselabels[1])
    ax[0].set_xlim([0,25])
    ax[0].set_xlabel("Specific Humidity [g/kg]")
    ax[1].plot(ds1['THETA'], ds1['plev']/100)
    ax[1].plot(ds2['THETA'], ds2['plev']/100)
    ax[1].set_xlabel("Potential Temperature [K]")
    ax[1].set_xlim([275, 325])
    ax[2].plot(ds1['CLDLIQ']*lscale, ds1['plev']/100)
    ax[2].plot(ds2['CLDLIQ']*lscale, ds2['plev']/100)
    ax[2].set_xlabel("Cloud Liquid [$\mu$g/kg]")
    ax[2].set_xlim([0,150])
    ax[2].invert_yaxis()
    ax[0].set_ylabel("Pressure")
    [a.spines['top'].set_visible(False) for a in ax]
    [a.spines['right'].set_visible(False) for a in ax]
    fig.legend(loc='upper left', bbox_to_anchor=(0.0, -0.01))
    return fig, ax


def process_case(climo_loc, latitude, longitude, pressurelevels):
    fils = sorted(list(Path(climo_loc).glob(f"*_T_*.nc")))
    temperature = xr.open_mfdataset(fils)['T'].sel(lat=latitude, lon=longitude).compute()
    fils = sorted(list(Path(climo_loc).glob(f"*_Q_*.nc")))
    vapor = xr.open_mfdataset(fils)['Q'].sel(lat=latitude, lon=longitude).compute()
    fils = sorted(list(Path(climo_loc).glob(f"*_CLDLIQ_*.nc")))
    liquid = xr.open_mfdataset(fils)['CLDLIQ'].sel(lat=latitude, lon=longitude).compute()
    # In one of these, we also need the hybrid-sigma coefficients
    fils = sorted(list(Path(climo_loc).glob(f"*_PS_*.nc")))
    ps_ds = xr.open_mfdataset(fils)
    ps = ps_ds['PS'].sel(lat=latitude, lon=longitude).compute()
    hyam = ps_ds['hyam'].isel(time=0).compute() # drop redundant time dimension
    hybm = ps_ds['hybm'].isel(time=0).compute()
    ps.name = "PS"
    # we aren't done. Interpolate to pressure levels here:
    t_plev = gc.interp_hybrid_to_pressure(temperature, ps, hyam, hybm, new_levels=pressurelevels, lev_dim='lev').compute()
    q_plev = gc.interp_hybrid_to_pressure(vapor, ps, hyam, hybm, new_levels=pressurelevels, lev_dim='lev').compute()
    liq_plev = gc.interp_hybrid_to_pressure(liquid, ps, hyam, hybm, new_levels=pressurelevels, lev_dim='lev').compute()
    t_plev.name = "T"
    q_plev.name = "Q"
    liq_plev.name = 'CLDLIQ'
    # But hold on, we actually want theta, not T:
    p = xr.DataArray(pressurelevels, dims='plev', coords={'plev': t_plev.plev})
    # pressurelevels expected in Pa!
    theta_plev = t_plev * ((100000. / p)**0.2854)  # https://glossary.ametsoc.org/wiki/Potential_temperature &  https://glossary.ametsoc.org/wiki/Poisson_constant
    # Still not done -- average over the area:
    w = np.cos(np.radians(temperature.lat))  # area weighting
    theta_aave = theta_plev.weighted(w).mean(dim=("lat","lon"))  # (12months x pressurelevels.shape[0])
    theta_aave.name = "THETA"
    q_aave = q_plev.weighted(w).mean(dim=("lat","lon"))
    liq_aave = liq_plev.weighted(w).mean(dim=("lat","lon"))
    ps_aave = ps.weighted(w).mean(dim=("lat","lon"))
    return xr.merge([theta_aave, q_aave, liq_aave, ps_aave])
