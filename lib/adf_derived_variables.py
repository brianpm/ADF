from pathlib import Path
import xarray as xr
import numpy as np
import numexpr as nex


def derive(expression, **inputs):
    """Derive output using expression, with input data suppled in inputs.

    expression : str
        A string describing the expression to evaluate, can contain
        numbers, variables, and basic operators. 

    inputs : dict
        A dictionary with keys being the variables used in expression
        and the values being the data (either xr.DataArray or numpy arrays)

    return
        The result of expression

    NOTES
    -----
        Example: expression = "PRECC + PRECL" 
                 inputs = {"PRECC": arr1, "PRECL": arr2}
    """
    return nex.evaluate(expression, inputs)  



def derive_xrWrap(expression, *args, **kwargs):

    # maybe this isn't kosher, but this allows the user
    # to either provide named arguments 
    # or a dictionary without using **dict
    if (len(args) == 1) and (isinstance(args[0], dict)):
        kwargs = args[0]

    template = kwargs[[*kwargs][0]] # whatever is the first input
    # pop the things that are not part of expression inputs
    dims = kwargs.pop("dims", (template.dims if hasattr(template, 'dims') else None))
    coords = kwargs.pop("coords", (template.coords if hasattr(template, 'coords') else None))
    attrs = kwargs.pop("attrs", template.attrs)
    ans = derive(expression, kwargs)
    return xr.DataArray(ans, dims=dims, coords=coords, attrs=attrs)



def derive_variables(self, vars_to_derive=None, ts_dir=None, overwrite=None):
    """
    Derive variables according to expression and inputs provided in vars_to_derive dict.

    Caution: this method _may still_ assume that there will be one time series file per variable

    If the file for the derived variable exists, the kwarg `overwrite` determines
    whether to overwrite the file (true) or exit with a warning message.
    """

    for var in vars_to_derive:
        constit_list = vars_to_derive[var][0]
        expression = vars_to_derive[var][1]
        # get constituent files:
        constit_files = {}
        for c in constit_list:
            if glob.glob(os.path.join(ts_dir, f"*.{c}.*")):
                constit_files[c] = sorted(glob.glob(os.path.join(ts_dir, f"*.{c}.*")))
            else:
                ermsg = f"{c} files were not present; {var} cannot be calculated."
                ermsg += f" Please remove {var} from diag_var_list or find the relevant CAM files."
                raise FileNotFoundError(ermsg)
        # from the constit_files, get the data
        constit_data = {}
        for c in constit_files:
            if len(constit_files[c] > 1):
                constit_data[c] = xr.open_mfdataset(constit_files[c])[c]
            else:
                constit_data[c] = xr.open_dataset(constit_files[c][0])[c]
        # output file specification:
        # TODO: this might not work for multi-file cases
        derived_file = constit_files[[*constit_files][0]].replace(constit_list[0], var)
        if Path(derived_file).is_file():
            if overwrite:
                Path(derived_file).unlink()
            else:
                print(
                    f"[{__name__}] Warning: {var} file was found and overwrite is False. Will use existing file."
                )
                continue
        # derive the variable using expression:
        result = derive_xrWrap(expression, **constit_data)  # defaults to copying metadata from 1st constituent
        # TODO: provide a way to send updated metadata to derive_xrWrap, maybe from additional variable_defaults info
        # Save output:
        if 'time' in result.dims:
            udim = 'time'
        else:
            udim = None
        result.to_netcdf(derived_file, unlimited_dims=udim, mode='w')