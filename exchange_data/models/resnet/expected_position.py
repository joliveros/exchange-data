import alog
import numpy as np
import pandas as pd


def expected_position_frame(df, **kwargs):
    df = df.copy()
    df.dropna(how='any', inplace=True)
    df = df.sort_index()
    df = df.reset_index(drop=False)

    flat_df = df[df['change'] == 0.0]
    short_df = df[df['change'] < 0.0]
    long_df = df[df['expected_position'] == 1]

    lens = [len(flat_df), len(short_df)]

    max_len = max(lens)

    flat_df = resample_to_len(flat_df, max_len)
    short_df = resample_to_len(short_df, max_len)
    flat_df = pd.concat([flat_df, short_df])

    max_len = max([len(flat_df), len(short_df)])

    long_df = resample_to_len(long_df, max_len)
    flat_df = resample_to_len(short_df, max_len)

    df = pd.concat((long_df, flat_df))

    return df


def resample_to_len(df, max_len):
    if len(df) < max_len:
        df = df.sample(frac=max_len / len(df),
                                 random_state=0,
                                 replace=True)
    return df
