from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl


# Calculate the end of month
def calc_end_of_month(date):
    return (date.replace(day=1) + timedelta(days=32)).replace(
        day=1,
    ) - timedelta(days=1)


# Calculate the end of year
def calc_end_of_year(date):
    return (date.replace(month=1, day=1) + timedelta(days=365)).replace(
        month=1,
        day=1,
    ) - timedelta(days=1)


# Test for duplicates
def test_for_duplicates(df, by_vec=None):
    if by_vec is None:
        return df.select(pl.col("*").is_duplicated()).sum()
    else:
        return df.select(pl.col(by_vec).is_duplicated()).sum()


# Create a vector of size k
def create_vec_of_size_k(k):
    return [np.nan] * k


# Calculate prior return sort vector
def calc_prior_ret_sort_vec(
    vec,
    only_non_na_bool=True,
    accept_nas=False,
    min_non_na=None,
    is_twelve_months=True,
):
    timeframe = 12 if is_twelve_months else 36

    if len(vec) <= timeframe:
        return create_vec_of_size_k(len(vec))
    else:
        retind = range(len(vec) - timeframe)
        pfformind = [i + timeframe for i in retind]

        meanvec2 = create_vec_of_size_k(len(vec))

        tosubtract = 2 if is_twelve_months else 13

        if only_non_na_bool:
            for i in retind:
                tmp = (
                    np.cumprod(
                        [1 + x for x in vec[i : i + timeframe - tosubtract]],
                    )
                    - 1
                )
                meanvec2[i + timeframe] = tmp[-1]
        elif accept_nas:
            for i in retind:
                if min_non_na is not None:
                    if (
                        np.sum(~np.isnan(vec[i : i + timeframe - tosubtract]))
                        >= min_non_na
                    ):
                        tmp = (
                            np.cumprod(
                                [
                                    1 + x
                                    for x in vec[i : i + timeframe - tosubtract]
                                    if not np.isnan(x)
                                ],
                            )
                            - 1
                        )
                        meanvec2[i + timeframe] = tmp[-1]
                    else:
                        meanvec2[i + timeframe] = np.nan
                else:
                    if any(~np.isnan(vec[i : i + timeframe - tosubtract])):
                        tmp = (
                            np.cumprod(
                                [
                                    1 + x
                                    for x in vec[i : i + timeframe - tosubtract]
                                    if not np.isnan(x)
                                ],
                            )
                            - 1
                        )
                        meanvec2[i + timeframe] = tmp[-1]
                    else:
                        meanvec2[i + timeframe] = np.nan
        return meanvec2


# Calculate prior return sort vector with variable timeframe
def calc_prior_ret_sort_vec_var_timeframe(
    vec,
    only_non_na_bool=True,
    accept_nas=False,
    min_non_na=None,
    from_t_minus=12,
    to_t_minus=2,
    assign_result_at_end_plus=0,
):
    window = from_t_minus - to_t_minus + 1

    if from_t_minus < to_t_minus:
        raise ValueError("Check input data")

    if len(vec) < 1 + from_t_minus + assign_result_at_end_plus:
        return create_vec_of_size_k(len(vec))
    else:
        retind = range(len(vec) - from_t_minus - assign_result_at_end_plus)
        pfformind = [
            i + from_t_minus + assign_result_at_end_plus for i in retind
        ]

        meanvec2 = create_vec_of_size_k(len(vec))

        if only_non_na_bool:
            for i in retind:
                tmp = np.cumprod([1 + x for x in vec[i : i + window - 1]]) - 1
                meanvec2[pfformind[i]] = tmp[-1]
        elif accept_nas:
            for i in retind:
                if min_non_na is not None:
                    if np.sum(~np.isnan(vec[i : i + window - 1])) >= min_non_na:
                        tmp = (
                            np.cumprod(
                                [
                                    1 + x
                                    for x in vec[i : i + window - 1]
                                    if not np.isnan(x)
                                ],
                            )
                            - 1
                        )
                        meanvec2[pfformind[i]] = tmp[-1]
                    else:
                        meanvec2[pfformind[i]] = np.nan
                else:
                    if any(~np.isnan(vec[i : i + window - 1])):
                        tmp = (
                            np.cumprod(
                                [
                                    1 + x
                                    for x in vec[i : i + window - 1]
                                    if not np.isnan(x)
                                ],
                            )
                            - 1
                        )
                        meanvec2[pfformind[i]] = tmp[-1]
                    else:
                        meanvec2[pfformind[i]] = np.nan
        return meanvec2


# Trim vector
def trim_vec(x, prob1=0.01, prob2=0.99):
    q1, q2 = np.nanquantile(x, [prob1, prob2])
    x = np.where(x < q1, np.nan, x)
    x = np.where(x > q2, np.nan, x)
    return x


# Winsorize vector
def winsorize_vec(x, prob1=0.01, prob2=0.99):
    q1, q2 = np.nanquantile(x, [prob1, prob2])
    x = np.where(x < q1, q1, x)
    x = np.where(x > q2, q2, x)
    return x


# Calculate log return from simple return
def calc_log_ret_from_simple_ret(x):
    if any(x <= -1):
        raise ValueError("RET < -1!")
    return np.log(1 + x)


# Calculate momentum vector for PFUS
def calc_mom_vec_pfus(
    vec,
    only_non_na_bool=True,
    accept_nas=False,
    min_non_na=None,
):
    if len(vec) <= 12:
        return create_vec_of_size_k(len(vec))
    else:
        retind = range(len(vec) - 12)
        pfformind = [i + 12 for i in retind]

        meanvec = create_vec_of_size_k(len(vec))

        if only_non_na_bool:
            for i in retind:
                if all(np.isfinite(vec[i : i + 11])):
                    meanvec[i + 12] = (
                        np.prod([1 + x for x in vec[i : i + 11]]) - 1
                    )
                else:
                    meanvec[i + 12] = np.nan
        elif accept_nas:
            for i in retind:
                if min_non_na is not None:
                    if np.sum(~np.isnan(vec[i : i + 11])) >= min_non_na:
                        tmp = (
                            np.cumprod(
                                [
                                    1 + x
                                    for x in vec[i : i + 11]
                                    if not np.isnan(x)
                                ],
                            )
                            - 1
                        )
                        meanvec[i + 12] = tmp[-1]
                    else:
                        meanvec[i + 12] = np.nan
                else:
                    if any(~np.isnan(vec[i : i + 11])):
                        tmp = (
                            np.cumprod(
                                [
                                    1 + x
                                    for x in vec[i : i + 11]
                                    if not np.isnan(x)
                                ],
                            )
                            - 1
                        )
                        meanvec[i + 12] = tmp[-1]
                    else:
                        meanvec[i + 12] = np.nan
        return meanvec


# Quick summary stats
def quick_summary_stats(vec):
    return {
        "mean": np.nanmean(vec),
        "std": np.nanstd(vec),
        "min": np.nanmin(vec),
        "percentiles": np.nanpercentile(
            vec,
            [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9],
        ),
        "max": np.nanmax(vec),
    }


# Read data
def read_data(csvfile, sep_new=",", header_new=True):
    return pl.read_csv(
        csvfile,
        separator=sep_new,
        has_header=header_new,
        null_values=["NA", "#WERT!", "#NV", ".."],
    )
