import alog
import click

from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex._long_position_emitter import \
    LongPositionEmitter


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
