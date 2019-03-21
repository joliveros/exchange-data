from abc import ABC

import click

from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex._bitmex_position_emitter import \
    BitmexPositionEmitter
from tgym.envs import LongOrderBookTradingEnv


class LongPositionEmitter(BitmexPositionEmitter, LongOrderBookTradingEnv, ABC):
    def __init__(self, **kwargs):
        # BitmexPositionEmitter.__init__(self, **kwargs)
        super().__init__(
            **kwargs
        )




@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__),
                default=BitmexChannels.XBTUSD.value)
@click.option('--job-name', '-n', default=None)
@click.option('--agent-cls', '-a', default=None)
@click.option('--checkpoint_id', '-c')
@click.option('--result-path', '-r')
def main(**kwargs):
    emitter = LongPositionEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    main()
