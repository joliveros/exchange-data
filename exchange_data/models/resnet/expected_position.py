import alog
import numpy as np
import pandas as pd


def expected_position_frame(df, take_ratio=1.0, **kwargs):
    df = df.copy()
    df.dropna(how='any', inplace=True)
    df = df.sort_index()
    df = df.reset_index(drop=False)

    # flat_df = df[df['expected_position'] == 0]
    long_df = df[df['expected_position'] == 1]

    flat_df = df[df['large_negative_change'] == 1]
    # flat_df = pd.concat([flat_df, negative_df.sample(frac=take_ratio, replace=True)])

    flat_df = resample_to_len(flat_df, int(len(long_df) * take_ratio))
    long_df = long_df.sample(frac=take_ratio, replace=True)

    # if take_ratio != 1.0:
    #     long_df = long_df.sample(frac=take_ratio, random_state=0, replace=True)

    df = pd.concat((long_df, flat_df))

    return df


def resample_to_len(df, max_len):
    len_df = len(df)
    if len_df > max_len:
        frac = max_len / len_df
    else:
        frac = len_df / max_len

    df = df.sample(frac=frac,
                   random_state=0,
                   replace=True)

    return df
