import click
from bitmex_websocket import Instrument
import websocket
import asyncio


@click.command()
@click.argument('symbol',
                required=True)
def main(symbol):
    """Saves bitmex data in realtime to influxdb"""
    print(symbol)
    websocket.enableTrace(True)

    XBTH17 = Instrument(symbol=symbol,
                        channels=['orderBookL2'],
                        # set to 1 because data will be saved to db
                        maxTableLength=1,
                        shouldAuth=False)

    XBTH17.on('action', lambda x: print("# action message: %s" % x))

    loop = asyncio.get_event_loop()
    return loop.run_forever()
