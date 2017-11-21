from . import Recorder
from . import settings
import asyncio
import click
import rollbar


rollbar.init(settings.ROLLBAR_API_KEY, 'production')


@click.command()
@click.argument('symbols',
                required=True)
@click.argument('https',
                default=False)
def main(symbols, https):
    """Saves bitmex data in realtime to influxdb"""
    Recorder(symbols)

    loop = asyncio.get_event_loop()
    return loop.run_forever()
