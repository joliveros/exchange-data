from . import BitmexRecorder
from .bitstamp import BitstampRecorder
from . import settings
from stringcase import pascalcase
import asyncio
import click
import rollbar
import save_bitmex_data


rollbar.init(settings.ROLLBAR_API_KEY, 'production')


@click.command()
@click.argument('symbol', nargs=-1, required=True)
@click.option('--exchange', required=True, help="an exchange such as bitstamp|bitmex")
@click.option('--https', default=False)
def main(symbol, exchange, https):
    recorder = getattr(save_bitmex_data, pascalcase(exchange) + 'Recorder')
    recorder(list(symbol))

    loop = asyncio.get_event_loop()
    return loop.run_forever()
