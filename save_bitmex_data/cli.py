from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)
from builtins import *
import click
from bitmex_websocket import Instrument
import websocket
import asyncio
from influxdb import InfluxDBClient
import os
from urllib.parse import urlparse


@click.command()
@click.argument('symbol',
                required=True)
def main(symbol):
    """Saves bitmex data in realtime to influxdb"""
    INFLUX_DB = os.environ.get('INFLUX_DB')
    conn_params = urlparse(INFLUX_DB)
    netlocs = conn_params.netloc.split(',')
    netloc = netlocs[0]
    parsed_netloc = _parse_netloc(netloc)
    db = InfluxDBClient(host=parsed_netloc['host'],
                        port=parsed_netloc['port'],
                        username=parsed_netloc['username'],
                        database=symbol,
                        ssl=True)
    # websocket.enableTrace(True)

    XBTH17 = Instrument(symbol=symbol,
                        channels=['instrument'],
                        # set to 1 because data will be saved to db
                        maxTableLength=1,
                        shouldAuth=False)

    XBTH17.on('instrument', on_table)

    loop = asyncio.get_event_loop()
    return loop.run_forever()


def on_table(table):
    print("#@")
    print(table)
    return


def _parse_netloc(netloc):
    info = urlparse("http://%s" % (netloc))
    return {'username': info.username or None,
            'password': info.password or None,
            'host': info.hostname or 'localhost',
            'port': info.port or 8086}
