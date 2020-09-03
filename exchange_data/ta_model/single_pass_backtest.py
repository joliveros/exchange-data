#!/usr/bin/env python

from exchange_data.ta_model.backtest import BackTest

import alog
import click
import pandas as pd


class SinglePassBackTest(BackTest):

    def test(self, **kwargs):
        df = self.label_position(**kwargs)

        self.capital = 1.0

        df.reset_index(drop=False, inplace=True)
        df['capital'] = self.capital
        df = df.apply(self.pnl, axis=1)

        # pd.set_option('display.max_rows', len(df) + 1)

        alog.info(df)

        if self.capital > 50.0:
            return 0.0

        return self.capital


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', '-m', default='4', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--session-limit', '-s', default=100, type=int)
@click.argument('symbol', type=str)
def main(**kwargs):
    SinglePassBackTest(**kwargs).test()


if __name__ == '__main__':
    main()
