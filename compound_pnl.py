#!/usr/bin/env python
import alog
import click
from pytimeparse.timeparse import timeparse


class CompoundPnl(object):
    def __init__(self, capital, interval, trade_interval='1h',
                 pnl_rate=97/9650):
        self.capital = capital
        self.interval = timeparse(interval)
        trade_interval = timeparse(trade_interval)
        self.steps = self.interval / trade_interval

        for step in range(0, int(self.steps)):
            # alog.info((1 + pnl_rate))
            # alog.info(self.capital * (1 + pnl_rate))
            # raise Exception()
            self.capital = self.capital * (1 + pnl_rate)

        alog.info(f'### {int(self.steps)} steps, capital: {self.capital}')



@click.command()
@click.option('--capital', '-c', type=float, default=1)
@click.option('--interval', '-i', type=str, default='1d')
def main(**kwargs):
    alog.info(alog.pformat(kwargs))

    CompoundPnl(**kwargs)


if __name__ == '__main__':
    main()
