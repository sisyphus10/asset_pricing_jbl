from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def get_data_dataframe(
    data_panel,
    return_panel,
    char_names,
    dates,
    permnos,
    quarterly_updates,
    mask,
):
    T, N, C = data_panel.shape
    if mask is None:
        nonnan_returns = np.argwhere(np.any(~np.isnan(data_panel), axis=2))
        num_nonnan_returns = np.sum(np.any(~np.isnan(data_panel), axis=2))
    else:
        nonnan_returns = np.argwhere(mask)
        num_nonnan_returns = np.sum(mask)

    data_matrix = np.zeros((num_nonnan_returns, C + 4))
    columns = np.append(char_names, ["date", "permno", "monthly_update"])
    for i in range(nonnan_returns.shape[0]):
        nonnan_return = nonnan_returns[i]
        data_matrix[i, :C] = data_panel[nonnan_return[0], nonnan_return[1], :]
        data_matrix[i, C] = return_panel[nonnan_return[0], nonnan_return[1]]
        data_matrix[i, C + 1] = dates[nonnan_return[0]]
        data_matrix[i, C + 2] = permnos[nonnan_return[1]]
        data_matrix[i, C + 3] = quarterly_updates[
            nonnan_return[0],
            nonnan_return[1],
        ]

    chars_and_returns_df = pd.DataFrame(data_matrix)
    chars_and_returns_df.columns = columns

    return chars_and_returns_df


def get_data_panel(path, computstat_data_present_filter=True, start_date=None):
    """
    fetch data from the path specified
    Parameters
    ----------
        path: location of data feather file
        computstat_data_present_filter: whether or not to apply the filter requiring stocks to have a compustat observation
        start_date: date to start data from
    """
    data = pd.read_feather(path)
    if start_date is not None:
        data = data.loc[data.date >= start_date]
    dates = data["date"].unique()
    dates.sort()
    permnos = data["permno"].unique().astype(int)
    permnos.sort()

    date_vals = [int(date) for date in dates]
    chars = np.array(data.columns.tolist()[:-4])
    chars.sort()

    percentile_rank_chars = np.zeros(
        (len(date_vals), permnos.shape[0], len(chars)),
    )
    percentile_rank_chars[:, :, :] = np.nan

    permno_map = np.zeros(int(max(permnos)) + 1, dtype=int)
    for i, permno in enumerate(permnos):
        permno_map[permno] = i

    for i, date in enumerate(dates):
        date_data = data.loc[data["date"] == date].sort_values(by="permno")
        date_permnos = date_data["permno"].to_numpy().astype(int)
        permno_inds_for_date = permno_map[date_permnos]
        percentile_rank_chars[i, permno_inds_for_date, :] = date_data[
            chars
        ].to_numpy()

    if computstat_data_present_filter:
        # cstat_permnos = pd.read_csv("/Users/johannes/Downloads/Research/Asset Pricing/missing_fin_data_code/data/compustat_permnos.csv")["PERMNO"].to_numpy()
        cstat_permnos = data["permno"].unique().astype(int)
        permno_filter = np.isin(permnos, cstat_permnos)
        percentile_rank_chars = percentile_rank_chars[:, permno_filter, :]
        permnos = permnos[permno_filter]

    return percentile_rank_chars, chars, date_vals, permnos


def get_data_panel2(df, computstat_data_present_filter=True, start_date=None):
    """
    fetch data from the path specified
    Parameters
    ----------
        path: location of data feather file
        computstat_data_present_filter: whether or not to apply the filter requiring stocks to have a compustat observation
        start_date: date to start data from
    """
    data = df
    if start_date is not None:
        data = data.loc[data.date >= start_date]
    dates = data["date"].unique()
    dates.sort()
    permnos = data["permno"].unique().astype(int)
    permnos.sort()

    date_vals = [int(date) for date in dates]

    # choose all non object columns as chars
    chars = data.select_dtypes(exclude=["object"]).columns.tolist()
    chars.remove("date")
    chars.remove("permno")
    chars.remove("age")
    chars.remove("exchg")

    # chars = np.array(data.columns.tolist()[:-4])  # TODO: Check which columns are the last for in Pelger
    chars.sort()

    percentile_rank_chars = np.zeros(
        (len(date_vals), permnos.shape[0], len(chars)),
    )
    percentile_rank_chars[:, :, :] = np.nan

    permno_map = np.zeros(int(max(permnos)) + 1, dtype=int)
    for i, permno in enumerate(permnos):
        permno_map[permno] = i

    for i, date in enumerate(dates):
        date_data = data.loc[data["date"] == date].sort_values(by="permno")
        date_data[chars] = date_data[chars].apply(
            pd.to_numeric,
            errors="coerce",
        )
        date_permnos = date_data["permno"].to_numpy().astype(int)
        permno_inds_for_date = permno_map[date_permnos]
        percentile_rank_chars[i, permno_inds_for_date, :] = date_data[
            chars
        ].to_numpy()

    if computstat_data_present_filter:
        # cstat_permnos = pd.read_csv("/Users/johannes/Downloads/Research/Asset Pricing/missing_fin_data_code/data/compustat_permnos.csv")["PERMNO"].to_numpy()
        cstat_permnos = data["permno"].unique().astype(int)
        permno_filter = np.isin(permnos, cstat_permnos)
        percentile_rank_chars = percentile_rank_chars[:, permno_filter, :]
        permnos = permnos[permno_filter]

    return percentile_rank_chars, chars, date_vals, permnos


CHAR_GROUPINGS = [
    ("A2ME", "Q"),
    ("AC", "Q"),
    ("AT", "Q"),
    ("ATO", "Q"),
    ("B2M", "QM"),
    ("BETA_d", "M"),
    ("BETA_m", "M"),
    ("C2A", "Q"),
    ("CF2B", "Q"),
    ("CF2P", "QM"),
    ("CTO", "Q"),
    ("D2A", "Q"),
    ("D2P", "M"),
    ("DPI2A", "Q"),
    ("E2P", "QM"),
    ("FC2Y", "QY"),
    ("IdioVol", "M"),
    ("INV", "Q"),
    ("LEV", "Q"),
    ("ME", "M"),
    ("TURN", "M"),
    ("NI", "Q"),
    ("NOA", "Q"),
    ("OA", "Q"),
    ("OL", "Q"),
    ("OP", "Q"),
    ("PCM", "Q"),
    ("PM", "Q"),
    ("PROF", "QY"),
    ("Q", "QM"),
    ("R2_1", "M"),
    ("R12_2", "M"),
    ("R12_7", "M"),
    ("R36_13", "M"),
    ("R60_13", "M"),
    ("HIGH52", "M"),
    ("RVAR", "M"),
    ("RNA", "Q"),
    ("ROA", "Q"),
    ("ROE", "Q"),
    ("S2P", "QM"),
    ("SGA2S", "Q"),
    ("SPREAD", "M"),
    ("SUV", "M"),
    ("VAR", "M"),
]
