from get_docker_secret import get_docker_secret
from os import environ

import alog
import logging
import rollbar

LOG_LEVEL = environ.get('LOG_LEVEL')
if LOG_LEVEL is None:
    LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.getLevelName(LOG_LEVEL)

alog.set_level(LOG_LEVEL)

alog.info(LOG_LEVEL)

RUN_ENV = environ.get('RUN_ENV')

DB = get_docker_secret('DB')

HOME = environ.get('HOME')

BITSTAMP_PUSHER_APP_KEY = get_docker_secret('BITSTAMP_PUSHER_APP_KEY')

BITMEX_API_KEY = get_docker_secret('BITMEX_API_KEY')
BITMEX_API_SECRET = get_docker_secret('BITMEX_API_SECRET')

if RUN_ENV != 'development':
    ROLLBAR_API_KEY = get_docker_secret('ROLLBAR_API_KEY')

    if ROLLBAR_API_KEY:
        rollbar.init(ROLLBAR_API_KEY, 'production')

MEASUREMENT_BATCH_SIZE = environ.get('BATCH_SIZE')

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)

if not MEASUREMENT_BATCH_SIZE:
    MEASUREMENT_BATCH_SIZE = 100
else:
    MEASUREMENT_BATCH_SIZE = int(MEASUREMENT_BATCH_SIZE)

alog.debug('batch size: %s' % MEASUREMENT_BATCH_SIZE)

CERT_FILE = './ca.pem'

REDIS_HOST = environ.get('REDIS_HOST') or 'redis'

WS_HOST = environ.get('WS_HOST') or 'proxy'

TICK_INTERVAL = '1s'

PROMETHEUS_HOST = environ.get('PROMETHEUS_HOST')

