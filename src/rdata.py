import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import globalenv, pandas2ri, r
from rpy2.robjects.packages import importr


def df_from_rdata(path, var_name):
    pandas2ri.activate()
    robj = r.load(path)
    with (ro.default_converter + pandas2ri.converter).context():
        data = ro.conversion.get_conversion().rpy2py(globalenv[var_name])
    return data

def csv_from_rdata(path_rdata, path_csv, var_name):
    data = df_from_rdata(path_rdata, var_name)
    data.to_csv(path_csv)
