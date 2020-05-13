#import statements:
import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

#Set non-X-window backend for matplotlib:
mpl.use('Agg')

#Now import pyplot:
import matplotlib.pyplot as plt

#################
#HELPER FUNCTIONS
#################

def use_this_norm():
    """Just use the right normalization; avoids a deprecation warning."""

    mplversion = [int(x) for x in mpl.__version__.split('.')]
    if mplversion[0] < 3:
        return mpl.colors.Normalize, mplversion[0]
    else:
        if mplversion[1] < 2:
            return mpl.colors.DivergingNorm, mplversion[0]
        else:
            return mpl.colors.TwoSlopeNorm, mplversion[0]

#######

def global_average(fld, wgt, verbose=False):
    """
    A simple, pure numpy global average.
    fld: an input ndarray
    wgt: a 1-dimensional array of weights
    wgt should be same size as one dimension of fld
    """

    s = fld.shape
    for i in range(len(s)):
        if np.size(fld, i) == len(wgt):
            a = i
            break
    fld2 = np.ma.masked_invalid(fld)
    if verbose:
        print("(global_average)-- fraction input missing: {}".format(fraction_nan(fld)))
        print("(global_average)-- fraction of mask that is True: {}".format(np.count_nonzero(fld2.mask) / np.size(fld2)))
        print("(global_average)-- apply ma.average along axis = {} // validate: {}".format(a, fld2.shape))
    avg1, sofw = np.ma.average(fld2, axis=a, weights=wgt, returned=True) # sofw is sum of weights

    return np.ma.average(avg1)

#######

def plot_map_and_save(wks, mdlfld, obsfld, diffld, **kwargs):
    """This plots mdlfld, obsfld, diffld in a 3-row panel plot of maps."""
    # preprocess
    # - assume all three fields have same lat/lon
    lat = obsfld['lat']
    wgt = np.cos(np.radians(lat))
    mwrap, lon = add_cyclic_point(mdlfld, coord=mdlfld['lon'])
    owrap, _ = add_cyclic_point(obsfld, coord=obsfld['lon'])
    dwrap, _ = add_cyclic_point(diffld, coord=diffld['lon'])
    wrap_fields = (mwrap, owrap, dwrap)
    # mesh for plots:
    lons, lats = np.meshgrid(lon, lat)
    # Note: using wrapped data makes spurious lines across plot (maybe coordinate dependent)
    lon2, lat2 = np.meshgrid(mdlfld['lon'], mdlfld['lat'])

    # get statistics (from non-wrapped)
    fields = (mdlfld, obsfld, diffld)
    area_avg = [global_average(x, wgt) for x in fields]

    d_rmse = np.sqrt(area_avg[-1] ** 2)  # RMSE

    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiString' in kwargs:
        tiString = kwargs.pop("tiString")
    else:
        tiString = ''
    if 'tiFontSize' in kwargs:
        tiFontSize = kwargs.pop('tiFontSize')
    else:
        tiFontSize = 8

    # We want to make sure mdlfld and obsfld are on the same contours:
    minval = np.min([np.min(mdlfld), np.min(obsfld)])
    maxval = np.max([np.max(mdlfld), np.max(obsfld)])
    normfunc, mplv = use_this_norm()
    if ((minval < 0) and (0 < maxval)) and mplv > 2:
        norm1 = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        cmap1 = 'coolwarm'
    else:
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        cmap1 = 'coolwarm'

    if 'cnLevels' in kwargs:
        levels1 = kwargs.pop(cnLevels)
    else:
        levels1 = np.linspace(minval, maxval, 12)
    # set a symmetric color bar for diff:
    absmaxdif = np.max(np.abs(diffld))
    # set levels for difference plot:
    levelsd = np.linspace(-1*absmaxdif, absmaxdif, 12)

    fig, ax = plt.subplots(figsize=(6,12), nrows=3, subplot_kw={"projection":ccrs.PlateCarree()})
    img = [] # contour plots
    cs = []  # contour lines
    cb = []  # color bars

    for i, a in enumerate(wrap_fields):
        if i == len(wrap_fields)-1:
            levels = levelsd #Using 'levels=12' casued len() error in mpl. -JN
            #Only use "vcenter" if "matplotlib" version is greater than 2:
            if(mplv > 2):
                norm = normfunc(vmin=-1*absmaxdif, vcenter=0., vmax=absmaxdif)
            else:
                norm = normfunc(vmin=-1*absmaxdif, vmax=absmaxdif)
            cmap = 'coolwarm'
        else:
            levels = levels1
            cmap = cmap1
            norm = norm1

        img.append(ax[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), **kwargs))
        cb.append(fig.colorbar(img[i], ax=ax[i], shrink=0.8))
        ax[i].set_title("AVG: {0:.3f}".format(area_avg[i]), loc='right', fontsize=tiFontSize)
        # add contour lines
        cs.append(ax[i].contour(lon2, lat2, fields[i], transform=ccrs.PlateCarree(), colors='k'))
        ax[i].clabel(cs[i], cs[i].levels, inline=True, fontsize=tiFontSize-2, fmt='%1.1f')
        ax[i].text( 10, -140, "CONTOUR FROM {} to {} by {}".format(min(cs[i].levels), max(cs[i].levels), cs[i].levels[1]-cs[i].levels[0]),
        bbox=dict(facecolor='none', edgecolor='black'), fontsize=tiFontSize-2)

    # set rmse title:
    ax[-1].set_title("RMSE: {0:.3f}".format(d_rmse), fontsize=tiFontSize)

    for a in ax:
        a.outline_patch.set_linewidth(2)
        a.coastlines()
        a.set_xticks(np.linspace(-180, 180, 7), crs=ccrs.PlateCarree())
        a.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        a.tick_params('both', length=10, width=2, which='major')
        a.tick_params('both', length=5, width=1, which='minor')
    fig.savefig(wks, bbox_inches='tight', dpi=300)

#
#  -- zonal mean code --
#

def pres_from_hybrid(psfc, hya, hyb, p0=100000.):
    # p = a(k)*p0 + b(k)*ps.
    return hya*p0 + hyb*psfc


def lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None):
    pressure = pres_from_hybrid(ps, hyam, hybm, P0)
    if new_levels is None:
        pnew = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]  # mandatory levels
    else:
        pnew = new_levels
    data_interp = data.interp(lev=pnew)
    return data_interp


def zonal_mean_xr(fld):
    """Average over all dimensions except `lev` and `lat`."""
    if isinstance(fld, xr.DataArray):
        d = fld.dims
        davgovr = [dim for dim in d if dim not in ('lev','lat')]
    else:
        raise IOError("zonal_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)


def zm_validate_dims(fld):
    # note: we can only handle variables that reduce to (lev, lat) or (lat,)
    if len(fld.dims) > 4:
        print(f"Sorry, too many dimensions: {fld.dims}")
        return None
    has_lev = 'lev' in fld.dims
    has_lat = 'lat' in fld.dims
    if not has_lat:
        return None
    else:
        return has_lat, has_lev


def _zonal_plot_line(ax, lat, data, **kwargs):
    ax.plot(lat, data, **kwargs)
    ax.set_xlim([max([lat.min(), -90.]), min([lat.max(), 90.])])
    return ax

def _zonal_plot_preslat(ax, lat, lev, data, **kwargs):
    mlev, mlat = np.meshgrid(lev, lat)
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'Spectral_r'
    img = ax.contourf(mlat, mlev, data.transpose('lat', "lev"), cmap=cmap, **kwargs)
    minor_locator = mpl.ticker.FixedLocator(lev)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(which='minor', length=4, color='r')
    ax.set_ylim([1000, 1])
    return img, ax

def zonal_plot(lat, data, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if 'lev' in data.dims:
        img, ax = _zonal_plot_preslat(ax, lat, data['lev'], data, **kwargs)
        return img, ax
    else:
        ax = _zonal_plot_line(ax, lat, data, **kwargs)
        return ax


def zonal_mean_plot(fld, lat=None, ax=None, ps=None, hya=None, hyb=None, **kwargs):
    if lat is None:
        lat = fld['lat']
    z = zonal_mean_xr(fld)
    if 'lev' in z.dims:
        print("FOUND LEV IN DIMS.")
        # where to get psurf and coefs? 
        if ps is None:
            ps = fld['PS']
        if hya is None:
            hya = fld['hyam']
        if hyb is None:
            hyb = fld['hybm']                   
        z = lev_to_plev(z, ps, hya, hyb)
    print(kwargs)
    A = zonal_plot(lat, z, ax=ax, **kwargs)
    f = plt.gcf()
    return f, A
    
    
def plot_zonal_mean_and_save(wks, adata, apsurf, ahya, ahyb, bdata, bpsurf, bhya, bhyb):
    """This is the default zonal mean plot:
        adata: data to plot ([lev], lat, [lon]) 
        apsurf: surface pressure (Pa) for adata when lev present; otherwise None
        ahya, ahyb: a and b hybrid-sigma coefficients when lev present; otherwise None
        same for b*.
        - For 2-d variables (reduced to (lat,)):
          + 2 panels: (top) zonal mean, (bottom) difference
        - For 3-D variables (reduced to (lev,lat)):
          + 3 panels: (top) zonal mean adata, (middle) zonal mean bdata, (bottom) diffdata
          + pcolormesh/contour plot
    """
    if apsurf is not None:
        aplev = lev_to_plev(adata, apsurf, ahya, ahyb, P0=100000., new_levels=None)
        bplev = lev_to_plev(bdata, bpsurf, bhya, bhyb, P0=100000., new_levels=None)
        azm = zonal_mean_xr(aplev)
        bzm = zonal_mean_xr(bplev)
        diff = azm - bzm
        fig, ax = plt.subplots(nrows=3, constrained_layout=True)
        zonal_plot(adata['lat'], azm, ax=ax[0])
        zonal_plot(bdata['lat'], bzm, ax=ax[1])
        zonal_plot(adata['lat'], diff, ax=ax[2])
    else:
        azm = zonal_mean_xr(adata)
        bzm = zonal_mean_xr(bdata)
        diff = azm - bzm
        fig, ax = plt.subplots(nrows=2, constrained_layout=True)
        zonal_plot(adata['lat'], azm, ax=ax[0])
        zonal_plot(bdata['lat'], bzm, ax=ax[0])
        zonal_plot(adata['lat'], diff, ax=ax[1])
    fig.savefig(wks, bbox_inches='tight', dpi=300)