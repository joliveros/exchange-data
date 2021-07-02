#!/usr/bin/env python
import alog
import click
from pytimeparse.timeparse import timeparse


class CompoundPnl(object):
    def __init__(self, capital, max_investment, interval, trade_interval='10m',
                 rate=5/9650):
        self.capital = capital
        self.interval = timeparse(interval)
        trade_interval = timeparse(trade_interval)
        self.steps = self.interval / trade_interval

        for step in range(0, int(self.steps)):
            # alog.info((1 + pnl_rate))
            # alog.info(self.capital * (1 + pnl_rate))
            # raise Exception()

            capital = self.capital

            if capital > max_investment:
                capital = max_investment

            self.capital -= capital

            alog.info(capital)

            self.capital += capital * (1 + rate)

        alog.info(f'### {int(self.steps)} steps, capital: {"${:,.2f}".format(self.capital)}')




@click.command()
@click.option('--capital', '-c', type=float, default=1)
@click.option('--rate', '-r', type=float, default=5/9650)
@click.option('--max-investment', '-m', type=float, default=1400)
@click.option('--interval', '-i', type=str, default='1d')
@click.option('--trade-interval', '-t', type=str, default='30m')
def main(**kwargs):
    CompoundPnl(**kwargs)


if __name__ == '__main__':
    main()
