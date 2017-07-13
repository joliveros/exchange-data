from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)
from bitmex_websocket import Instrument
from builtins import *
from datetime import datetime
from influxdb import InfluxDBClient
from urllib.parse import urlparse
import alog
import asyncio
import click
import json
import os
import websocket
import rollbar


RUN_ENV = os.environ.get('RUN_ENV')

if RUN_ENV != 'development':
    ROLLBAR_API_KEY = os.environ.get('ROLLBAR_API_KEY')
    if not ROLLBAR_API_KEY:
        raise ValueError('Rollbar api key is required for production.')
    alog.info('rollbar initialized...')
    rollbar.init(ROLLBAR_API_KEY, 'production')

LOG_LEVEL = os.environ.get('LOG_LEVEL')
MEASUREMENT_BATCH_SIZE = os.environ.get('BATCH_SIZE')

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)
if not MEASUREMENT_BATCH_SIZE:
    MEASUREMENT_BATCH_SIZE = 100
else:
    MEASUREMENT_BATCH_SIZE = int(MEASUREMENT_BATCH_SIZE)

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)

db = {}
measurements = []

@click.command()
@click.argument('symbols',
                required=True)
def main(symbols):
    """Saves bitmex data in realtime to influxdb"""
    global db
    INFLUX_DB = os.environ.get('INFLUX_DB')
    LOG_LEVEL = os.environ.get('LOG_LEVEL')
    if LOG_LEVEL:
        alog.set_level(LOG_LEVEL)
    alog.debug(INFLUX_DB)
    CERT_FILE = './ca.pem'
    conn_params = urlparse(INFLUX_DB)
    database = conn_params.path[1:]
    netlocs = conn_params.netloc.split(',')
    netloc = netlocs[0]
    parsed_netloc = _parse_netloc(netloc)

    db = InfluxDBClient(host=parsed_netloc['host'],
                        port=parsed_netloc['port'],
                        username=parsed_netloc['username'],
                        password=parsed_netloc['password'],
                        database=database,
                        ssl=True,
                        verify_ssl=CERT_FILE)

    websocket.enableTrace(os.environ.get('RUN_ENV') == 'development')


    channels = [
                'trade',
                'quote',
                'orderBookL2'
                ]

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
        instrument.on('action', on_action)

    loop = asyncio.get_event_loop()
    return loop.run_forever()

def _parse_netloc(netloc):
    info = urlparse("http://%s" % (netloc))
    return {'username': info.username or None,
            'password': info.password or None,
            'host': info.hostname or 'localhost',
            'port': info.port or 8086}

def on_table(table_name, table):
    if table_name == 'trade':
        return on_trade(table)
    elif table_name == 'quote':
        return on_quote(table)



def on_action(message):
    table = message['table']

    if table == 'orderBookL2':
        data = message.copy()
        data = to_lowercase_keys(data)
        data['symbol'] = data['data'][0]['symbol']
        data['timestamp'] = get_timestamp()
        data.pop('table', None)

        for row in data['data']:
            row.pop('symbol', None)

        data = values_to_str(data)
        save_measurement(table, data)

def pp(data):
    alog.debug(json.dumps(data, indent=2, sort_keys=True))

def get_timestamp():
    return f'{str(datetime.utcnow())}Z'

def on_quote(table):
    data = table.copy()
    data = to_lowercase_keys(data)
    data['bidsize'] = str(data['bidsize'])
    data['bidprice'] = str(data['bidprice'])
    data['asksize'] = str(data['asksize'])
    data['askprice'] = str(data['askprice'])
    data = values_to_str(data)
    save_measurement('quote', data)

def on_trade(table):
    data = table.copy()
    data = to_lowercase_keys(data)
    data.pop('homenotional', None)
    data.pop('foreignnotional', None)
    data.pop('grossvalue', None)
    data['price'] = float(data['price'])
    data = values_to_str(data)
    save_measurement('trade', data)

def to_lowercase_keys(data):
    return dict((k.lower(), v) for k,v in data.items())

def values_to_str(data):
    keys = data.keys()

    for key in keys:
        value = data[key]
        if isinstance(value, (dict, list)):
            data[key] = json.dumps(value)

    return data

def save_measurement(name, table):
    global measurements
    global db

    timestamp = table.pop('timestamp', None)
    symbol = table.pop('symbol', None)

    measurement = {
        'measurement': name,
        'tags': {
            'symbol': symbol
        },
        'time': timestamp,
        'fields': table
    }

    pp(measurement)
    measurements.append(measurement)

    if len(measurements) >= MEASUREMENT_BATCH_SIZE:
        alog.debug('### Save measurements count: %s' % len(measurements))
        alog.debug(measurements)
        db.write_points(measurements, time_precision='ms')

        measurements = []
