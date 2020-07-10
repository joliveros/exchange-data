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

    if len(flat_df) > len(long_df):
        df = long_df.copy()
        long_df = pd.concat((df.sample(len(flat_df) - len(long_df),
                                            random_state=0, replace=True),
                             long_df))
    else:
        df = flat_df.copy()
        flat_df = pd.concat((df.sample(len(long_df) - len(flat_df),
                                           random_state=0, replace=True),
                             flat_df))

    if take_ratio != 1.0:
        long_df = long_df.sample(frac=take_ratio, replace=True, random_state=0)

    alog.info('#### here ####')

    alog.info((len(flat_df), len(long_df)))

    alog.info('#### here ####')

    df = pd.concat((long_df, flat_df))
    return df
