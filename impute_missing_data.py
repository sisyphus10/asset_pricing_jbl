from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyreadr

from imputation import core_imputation_model
from imputation import core_utils

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
# pelgers data

data_path = "/Users/johannes/Downloads/Research/Asset Pricing/missing_fin_data_code/data/example_data.fthr"
percentile_rank_chars, chars, date_vals, permnos = core_utils.get_data_panel(
    path=data_path,
    computstat_data_present_filter=True,
    start_date=19770000,
)


# %%
file_path = "/Users/johannes/Dropbox/ML AP Project/empirical analysis/data/gfin_data_yly_and_mthly_global_fin_p1.RData"
df = pyreadr.read_r(file_path)
df = df["cstatmerged_p1"]


# %%
# substitute all -inf and inf values with nan
df = df.replace([np.inf, -np.inf], np.nan)

# %%
# Convert data in required format

date_columns = ["date", "datadate_yly", "beg4mgap", "end4mgap"]
df[date_columns] = df[date_columns].apply(pd.to_datetime, errors="coerce")

# convert int columns defined as object to int
df["age"] = df["age"].astype("Int64")
df["exchg"] = df["exchg"].astype("Int64")

# %%
# change gvkey to permo to match pelgers notation
df = df.rename(columns={"gvkey": "permno"})
df["date"] = df["date"].dt.strftime("%Y%m%d").astype(int)

# drop these columns datadate_yly', 'beg4mgap', 'end4mgap
df = df.drop(columns=["datadate_yly", "beg4mgap", "end4mgap"])

# %%
# store all float type columns in a list
float_columns = df.select_dtypes(include=["float64"]).columns.tolist()
# add permno and date to the list
float_columns.extend(["permno", "date"])
# subset df to only include float columns
df = df[float_columns]

# %%
# exlude all columns from df which have more than 50% nan values
df = df.dropna(thresh=0.9 * len(df), axis=1)

# %%
# select only observations from 20000000 onwards
df = df[df["date"] >= 20000000]
# %%
df_dtypes = df.dtypes.to_frame()
# %%
percentile_rank_chars, chars, date_vals, permnos = core_utils.get_data_panel2(
    df=df,
    computstat_data_present_filter=True,
    start_date=19770000,
)

# TODO: CHeck percentile_rank_chars and if there are too many nan, e.g. columns only with nan
# %%
T, N, L = percentile_rank_chars.shape

# %%
dtypes = percentile_rank_chars.dtype

# %%
imputation = core_imputation_model.impute_data_xs(
    percentile_rank_chars,
    # n_xs_factors=20,
    n_xs_factors=4,
    time_varying_loadings=True,
    xs_factor_reg=0.01 / L,
    min_xs_obs=1,
)
# %%
