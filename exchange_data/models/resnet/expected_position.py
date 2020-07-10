import alog
import numpy as np
import pandas as pd


def expected_position_frame(df, expected_position_length, take_ratio=1.0,
                            pos_change_min=0.6, pos_change_max=0.6):
    df = df.copy()
    df.dropna(how='any', inplace=True)
    df = df.sort_index()
    df = df.reset_index(drop=False)

    change_df = df['best_ask'] - df['best_ask'].shift(1)
    change_df.dropna(how='any', inplace=True)
    change = change_df.to_numpy()
    pos_change = change[change > 0.0]
    min_change = np.quantile(pos_change, pos_change_min)
    max_change = np.quantile(pos_change, pos_change_max)

    change = df['best_ask'] - df['best_ask'].shift(1)

    df['expected_position'] = np.where(change >= min_change & change <=
                                       max_change, 1, 0)

    if expected_position_length > 1:
        for i in range(0, len(df)):
            expected_position = df.loc[i, 'expected_position'] == 1

            if expected_position:
                for x in range(0, expected_position_length):
                    y = i - x
                    if y >= 0:
                        df.loc[y, 'expected_position'] = 1

    df.astype({'expected_position': 'int32'})
    df.dropna(how='any', inplace=True)

    # for i in range(0, len(df)):
    #     row = df.loc[i]
    #     alog.info(row)
    #
    # alog.info(df)
    #
    # raise Exception()

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
