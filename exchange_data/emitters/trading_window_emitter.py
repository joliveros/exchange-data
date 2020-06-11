from exchange_data import Database
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils

import click


class TradingWindowEmitter(Messenger, Database):
    def __init__(
        self,
        symbol=BitmexChannels.XBTUSD,
        **kwargs
    ):
        super().__init__(
            database_batch_size=1,
            database_name='bitmex',
            **kwargs
        )

        self.symbol = symbol
        self.channels += ['2s']

        self.on('2s', self.publish_to_channels)

    def run(self):
        self.sub(self.channels)


@click.command()
def main(**kwargs):
    record = TradingWindowEmitter(**kwargs)
    record.run()


if __name__ == '__main__':
    main()
