from datetime import datetime
from dateutil.tz import tz
from exchange_data.utils import DateTimeUtils

import alog
import re
import time


class BinanceUtils(object):
    def sleep_during_embargo(self, e):
        if e.status_code == 418:
            embargo_timestamp = int(re.search('\d+', e.message)[0])
            embargo_timestamp = datetime.fromtimestamp(embargo_timestamp / 1000) \
                .replace(tzinfo=tz.tzlocal())
            alog.info(f'banned until {embargo_timestamp}')
            diff = embargo_timestamp - DateTimeUtils.now()

            sleep_seconds = diff.total_seconds()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
