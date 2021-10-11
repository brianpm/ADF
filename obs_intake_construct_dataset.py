import ast
import intake   # Need to install intake AND ALSO intake-esm



def load_catalog():
    # Catalog constructed by Max Grover:
    obs_catalog = intake.open_esm_datastore(
        "/glade/work/mgrover/intake-esm-catalogs/amwg_obs_datasets.json",
        csv_kwargs={"converters": {"variables": ast.literal_eval}},
        sep="/")
    return obs_catalog


def generate_obs_dataset(catalog, variables, sources=None):
    catalog_subset = catalog.search(variables=variables)
    return catalog_subset.to_dataset_dict()
    


if __name__ == "__main__":
    list_of_variables = ["T","SST","PRECT"]
    list_of_sources = None
    cat = load_catalog()
    dses = generate_obs_dataset(cat, list_of_variables, sources=list_of_sources)
    print(dses)