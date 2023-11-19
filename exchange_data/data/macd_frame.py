#!/usr/bin/env python
from plotly.subplots import make_subplots

from exchange_data.data.measurement_frame import MeasurementFrame
from pandas import DataFrame
from ta.trend import MACD

import numpy as np
import alog
import click
import pandas as pd

pd.options.plotting.backend = "plotly"


class MacdFrame(MeasurementFrame):
    def __init__(
        self,
        symbol,
        window_slow=26,
        window_fast=12,
        window_sign=9,
        window_factor=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.symbol = symbol
        self.window_slow = window_slow * window_factor
        self.window_fast = window_fast * window_factor
        self.window_sign = window_sign * window_factor

    @property
    def name(self):
        return f"{self.symbol}_OrderBookFrame"
        # return re.sub(r'(?<!^)(?=[A-Z])', '_', type(self).__name__).lower()

    @property
    def frame(self):
        query = (
            f"SELECT last(best_bid) AS data FROM {self.name} WHERE time >="
            f" {self.formatted_start_date} AND "
            f"time <= {self.formatted_end_date} GROUP BY time("
            f"{self.group_by})"
        )

        alog.info(query)
        frames = []

        for data in self.query(query).get_points(self.name):
            timestamp = self.parse_db_timestamp(data["time"])

            data = dict(time=timestamp, price=data["data"])

            frames.append(data)

        df = DataFrame.from_dict(frames)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        alog.info(df)

        df = df["price"].resample("10MIN").ohlc()
        df["macd_diff"] = MACD(
            close=df["close"],
            window_slow=self.window_slow,
            window_fast=self.window_fast,
            window_sign=self.window_sign,
        ).macd_diff()
        df["macd_diff"].fillna(0, inplace=True)

        df["trade"] = np.where(df["macd_diff"] >= 0, 0, 1)

        return df


@click.command()
@click.option("--database_name", "-d", default="binance", type=str)
@click.option("--group-by", "-g", default="15s", type=str)
@click.option("--interval", "-i", default="3h", type=str)
@click.option("--plot", "-p", is_flag=True)
@click.argument("symbol", type=str)
def main(**kwargs):
    df = MacdFrame(**kwargs).frame

    pd.set_option("display.max_rows", len(df) + 1)

    alog.info(df)


if __name__ == "__main__":
    main()
