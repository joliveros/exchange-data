from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex._bitmex_position_emitter import BitmexPositionEmitter
from exchange_data.trading import Positions
from tgym.envs.orderbook import LongOrderBookTradingEnv

import alog
import click


class LongPositionEmitter(BitmexPositionEmitter, LongOrderBookTradingEnv):
    def __init__(self, **kwargs):
        LongOrderBookTradingEnv.__init__(self, is_training=False, **kwargs)
        BitmexPositionEmitter.__init__(self, env='long-orderbook-trading-v0',
                                       **kwargs)

    def publish_position(self, action):
        _action = None

        if Positions.Flat.value == action:
            _action = Positions.Flat
        elif Positions.Long.value == action:
            _action = Positions.Long

        if _action:
            self.publish(self.job_name, dict(data=_action.name))



@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__),
                default=BitmexChannels.XBTUSD.value)
@click.option('--job-name', '-n', default=None)
@click.option('--agent-cls', '-a', default=None)
@click.option('--checkpoint_id', '-c')
@click.option('--result-path', '-r')
def main(**kwargs):
    emitter = LongPositionEmitter(**kwargs)
    alog.info(emitter.action_space)
    emitter.start()


if __name__ == '__main__':
    main()
