import pathlib
import xarray as xr
import traceback
from ecgtools import Builder
from ecgtools.builder import INVALID_ASSET, TRACEBACK

# Build a parser using ecgtools
def parse_amwg_obs(file):
    """Atmospheric observational data stored in"""
    file = pathlib.Path(file)
    info = {}

    try:
        stem = file.stem
        split = stem.split('_')
        source = split[0]
        temporal = split[-2]
        if len(temporal) == 2:
            month_number = int(temporal)
            time_period = 'monthly'
            temporal = datetime(2020, month_number, 1).strftime('%b').upper()

        elif temporal == 'ANN':
            time_period = 'annual'
        else:
            time_period = 'seasonal'

        with xr.open_dataset(file, chunks={}, decode_times=False) as ds:
            variable_list = [var for var in ds if 'long_name' in ds[var].attrs]

            info = {
                'source': source,
                'temporal': temporal,
                'time_period': time_period,
                'variable': variable_list,
                'path': str(file),
            }

        return info

    except Exception:
        return {INVALID_ASSET: file, TRACEBACK: traceback.format_exc()}


#
# Where are the files:
#

data_location = '/glade/p/cesm/amwg/amwg_diagnostics/obs_data'

# -- Example --
# files = sorted(glob.glob('/glade/p/cesm/amwg/amwg_diagnostics/obs_data/*'))
# info = parse_amwg_obs(files[0])
# print(info)

# Use ecgtools to build the catalog
b = Builder(data_location)

b.build(parse_amwg_obs)

cat_file = "/glade/work/brianpm/intake-esm-catalogs/amwg_obs_datasets.csv"

b.save(
    # File path - could save as .csv (uncompressed csv) or .csv.gz (compressed csv)
    cat_file,
    # Column name including filepath
    path_column_name='path',
    # Column name including variables
    variable_column_name='variables',
    # Data file format - could be netcdf or zarr (in this case, netcdf)
    data_format="netcdf",
    # Which attributes to groupby when reading in variables using intake-esm
    groupby_attrs=["source", "time_period"],
    # Aggregations which are fed into xarray when reading in data using intake
    aggregations=[
        {
            'type': 'join_new',
            'attribute_name': 'temporal',
            'options': {'coords': 'minimal', 'compat': 'override'},
        },
    ],
)

print(f"COMPLETE: Catalog should be saved as: {cat_file}")