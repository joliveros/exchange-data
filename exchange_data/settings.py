import logging

import rollbar
from os import environ
import alog

RUN_ENV = environ.get('RUN_ENV')

DB = environ.get('DB')
HOME = environ.get('HOME')

BITSTAMP_PUSHER_APP_KEY = environ.get('BITSTAMP_PUSHER_APP_KEY')

if RUN_ENV != 'development':
    ROLLBAR_API_KEY = environ.get('ROLLBAR_API_KEY')

    if ROLLBAR_API_KEY:
        rollbar.init(ROLLBAR_API_KEY, 'production')

MEASUREMENT_BATCH_SIZE = environ.get('BATCH_SIZE')

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)
if not MEASUREMENT_BATCH_SIZE:
    MEASUREMENT_BATCH_SIZE = 100
else:
    MEASUREMENT_BATCH_SIZE = int(MEASUREMENT_BATCH_SIZE)

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)


LOG_LEVEL = environ.get('LOG_LEVEL')

if LOG_LEVEL is None:
    LOG_LEVEL = logging.INFO

LOG_LEVEL = logging.getLevelName(LOG_LEVEL)

alog.debug(LOG_LEVEL)

alog.set_level(LOG_LEVEL)

CERT_FILE = './ca.pem'

REDIS_HOST = 'redis'
