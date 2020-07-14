import alog
import numpy as np
import pandas as pd


def expected_position_frame(df, take_ratio=1.0, **kwargs):
    df = df.copy()
    df.dropna(how='any', inplace=True)
    df = df.sort_index()
    df = df.reset_index(drop=False)

    flat_df = df[df['expected_position'] == 0]
    long_df = df[df['expected_position'] == 1]
    negative_df = df[df['large_negative_change'] == 1]

    max_len = max([len(flat_df), len(long_df), len(negative_df)])

    flat_df = resample_to_len(flat_df, max_len)

    long_df = resample_to_len(long_df, max_len)

    negative_df = resample_to_len(negative_df, max_len)

    if take_ratio != 1.0:
        long_df = long_df.sample(frac=take_ratio, random_state=0, replace=True)

    # flat_df = resample_to_len(short_df, max_len)

    df = pd.concat((long_df, flat_df, negative_df))

    return df


def resample_to_len(df, max_len):
    if len(df) < max_len:
        df = df.sample(frac=max_len / len(df),
                                 random_state=0,
                                 replace=True)
    return df
