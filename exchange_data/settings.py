import logging

import rollbar
from dotenv import load_dotenv, find_dotenv
from os import environ
import alog

load_dotenv(find_dotenv(), override=True)

RUN_ENV = environ.get('RUN_ENV')
DB = environ.get('DB')
BITSTAMP_PUSHER_APP_KEY = environ.get('BITSTAMP_PUSHER_APP_KEY')

if RUN_ENV != 'development':
    ROLLBAR_API_KEY = environ.get('ROLLBAR_API_KEY')
    rollbar.init(ROLLBAR_API_KEY, 'production')

    if not ROLLBAR_API_KEY:
        raise ValueError('Rollbar api key is required for production.')

MEASUREMENT_BATCH_SIZE = environ.get('BATCH_SIZE')

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)
if not MEASUREMENT_BATCH_SIZE:
    MEASUREMENT_BATCH_SIZE = 100
else:
    MEASUREMENT_BATCH_SIZE = int(MEASUREMENT_BATCH_SIZE)

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)


INFLUX_DB = environ.get('INFLUX_DB')
LOG_LEVEL = environ.get('LOG_LEVEL')

if LOG_LEVEL is None:
    LOG_LEVEL = logging.INFO

LOG_LEVEL = logging.getLevelName(LOG_LEVEL)

alog.debug(LOG_LEVEL)

alog.set_level(LOG_LEVEL)

alog.debug(INFLUX_DB)

CERT_FILE = './ca.pem'
