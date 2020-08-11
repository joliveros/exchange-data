#!/usr/bin/env python
from cached_property import cached_property
from exchange_data.data.measurement_frame import MeasurementFrame
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

import alog
import click
import pandas as pd


pd.options.plotting.backend = 'plotly'


class PriceFrame(MeasurementFrame):
    positive_change_count = 0
    min_consecutive_count = 1

    def __init__(
        self,
        database_name,
        interval,
        symbol,
        window_size,
        **kwargs
    ):
        super().__init__(
            batch_size=1,
            database_name=database_name,
            interval=interval,
            symbol=symbol,
            **kwargs)

        self.group_by_delta = pd.Timedelta(seconds=timeparse(self.group_by))
        self.window_size = window_size
        self.symbol = symbol

    @property
    def name(self):
        return f'{self.symbol}_OrderBookFrame_depth_40'

    @cached_property
    def frame(self):
        query = f'SELECT first(best_ask) AS best_ask, first(best_bid) AS ' \
                f'best_bid FROM {self.name} WHERE time ' \
                f'>=' \
                f' {self.formatted_start_date} AND ' \
                f'time <= {self.formatted_end_date} GROUP BY time(' \
                f'{self.group_by})'

        alog.info(query)

        frames = []

        for data in self.query(query).get_points(self.name):
            data['time'] = self.parse_db_timestamp(data['time'])

            frames.append(data)

        df = DataFrame.from_dict(frames)

        df['time'] = pd.to_datetime(df['time'])

        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        return df



@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--tick', is_flag=True)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    price = PriceFrame(**kwargs)

    # pd.set_option('display.max_rows', len(df) + 1)

    alog.info(price.frame)



if __name__ == '__main__':
    main()
