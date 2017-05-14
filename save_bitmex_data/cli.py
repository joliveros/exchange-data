from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)
from builtins import *
from bitmex_websocket import Instrument
from influxdb import InfluxDBClient
from urllib.parse import urlparse
import asyncio
import click
import os
import websocket
import json

MEASUREMENT_BATCH_SIZE = os.environ.get('BATCH_SIZE') || 100
db = {}
measurements = []

@click.command()
@click.argument('symbols',
                required=True)
def main(symbols):
    """Saves bitmex data in realtime to influxdb"""
    global db
    INFLUX_DB = os.environ.get('INFLUX_DB')
    CERT_FILE = './cert.ca'
    conn_params = urlparse(INFLUX_DB)
    netlocs = conn_params.netloc.split(',')
    netloc = netlocs[0]
    parsed_netloc = _parse_netloc(netloc)

    db = InfluxDBClient(host=parsed_netloc['host'],
                        port=parsed_netloc['port'],
                        username=parsed_netloc['username'],
                        password=parsed_netloc['password'],
                        database='bitmex',
                        ssl=True,
                        verify_ssl=CERT_FILE)

    websocket.enableTrace(os.environ.get('RUN_ENV') == 'development')


    channels = ['quote', 'trade']

    symbols = [x.strip() for x in symbols.split(',')]

    for symbol in symbols:
        instrument = Instrument(symbol=symbol,
                            channels=channels,
                            # set to 1 because data will be saved to db
                            maxTableLength=1,
                            shouldAuth=False)
        for table in channels:
            instrument.on(table, on_table)

        instrument.on('latency', lambda latency: print("latency: {0}".format(latency)))

    loop = asyncio.get_event_loop()
    return loop.run_forever()

def _parse_netloc(netloc):
    info = urlparse("http://%s" % (netloc))
    return {'username': info.username or None,
            'password': info.password or None,
            'host': info.hostname or 'localhost',
            'port': info.port or 8086}


def on_table(table_name, table):
        global measurements
        global db
        # print(json.dumps(table, indent=4, sort_keys=True))
        timestamp = table.pop('timestamp', None)
        symbol = table.pop('symbol', None)

        for key in table.keys():
            lower_key = key.lower()
            if 'price' in lower_key and table[key] is not None:
                table[key] = float(table[key])
            if 'foreignnotional' == lower_key:
                table[key] = float(table[key])
            if 'homenotional' == lower_key:
                table[key] = float(table[key])

        measurement = {
            'measurement': table_name,
            'tags': {
                'symbol': symbol
            },
            'time': timestamp,
            'fields': table
        }

        measurements.append(measurement)

        if len(measurements) > MEASUREMENT_BATCH_SIZE - 1:
            db.write_points(measurements, time_precision='ms')
            # print(json.dumps(measurements, indent=4, sort_keys=True))
            measurements = []
