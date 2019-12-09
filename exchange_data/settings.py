from os import environ

import alog
import logging
import rollbar
import sec

LOG_LEVEL = environ.get('LOG_LEVEL')
if LOG_LEVEL is None:
    LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.getLevelName(LOG_LEVEL)

alog.set_level(LOG_LEVEL)
RUN_ENV = environ.get('RUN_ENV')

DB = sec.load('DB', lowercase=False)

alog.debug(f'## db conn {DB} ##')

HOME = environ.get('HOME')

BITSTAMP_PUSHER_APP_KEY = sec.load('BITSTAMP_PUSHER_APP_KEY', lowercase=False)

BITMEX_API_KEY = sec.load('BITMEX_API_KEY', lowercase=False)
BITMEX_API_SECRET = sec.load('BITMEX_API_SECRET', lowercase=False)

if RUN_ENV != 'development':
    ROLLBAR_API_KEY = sec.load('ROLLBAR_API_KEY', lowercase=False)

    if ROLLBAR_API_KEY:
        rollbar.init(ROLLBAR_API_KEY, 'production')

MEASUREMENT_BATCH_SIZE = environ.get('BATCH_SIZE')

if not MEASUREMENT_BATCH_SIZE:
    MEASUREMENT_BATCH_SIZE = 100
else:
    MEASUREMENT_BATCH_SIZE = int(MEASUREMENT_BATCH_SIZE)

CERT_FILE = './ca.pem'

REDIS_HOST = environ.get('REDIS_HOST') or 'redis'

WS_HOST = environ.get('WS_HOST') or 'proxy'

TICK_INTERVAL = '1s'

PROMETHEUS_HOST = environ.get('PROMETHEUS_HOST')

