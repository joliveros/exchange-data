from exchange_data.emitters import SignalInterceptor
from exchange_data.emitters.messenger import Messenger
from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook.utils import Positions

import click


class TradeExecutor(
    Messenger,
    DateTimeUtils,
    SignalInterceptor
):

    def __init__(self, algo_channel: str, **kwargs):
        super().__init__(exit_func=self.stop, **kwargs)
        self.algo_channel = algo_channel

    def start(self):
        self.sub([Positions.Flat, Positions.Long, Positions.Short])


@click.command()
@click.argument(
    'algo_channel',
    type=str,
    default=None,
    help='Channel to listen for trade commands.'
)
def main(**kwargs):
    time_emitter = TradeExecutor(**kwargs)
    time_emitter.start()


if __name__ == '__main__':
    main()
