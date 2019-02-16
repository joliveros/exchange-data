from datetime import datetime, timezone, timedelta

from exchange_data import settings
from exchange_data.emitters.messenger import Messenger
from pytimeparse.timeparse import timeparse
from time import sleep

import alog
import click

from exchange_data.utils import NoValue


class TimeEmitter(Messenger):

    def __init__(self, tick_interval: str = settings.TICK_INTERVAL):
        super().__init__()
        self.tick_interval = timeparse(tick_interval)
        self.padding = 1100
        self.previous_day = self.next_day
        self.minute_counter = 0

    @property
    def next_tick(self):
        now = datetime.now()
        second_diff = (now.microsecond + self.padding)/999999
        next_tick = self.tick_interval - second_diff

        if next_tick > 0:
            return next_tick
        else:
            return 0

    @property
    def next_day(self):
        now: datetime = self.now_utc()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_date = today + timedelta(days=1)

        next_day_timestamp = next_date.timestamp() * 1000

        return next_day_timestamp

    @staticmethod
    def timestamp():
        now = TimeEmitter.now_utc().timestamp() * 1000
        return now

    @staticmethod
    def now_utc():
        return datetime.now().replace(tzinfo=timezone.utc)

    def start(self):
        while True:
            sleep(1)
            now = self.timestamp()

            self.minute_counter += 1
            if self.minute_counter % 60 == 0:
                self.publish('1m', str(now))
                self.minute_counter = 0

            self.publish(TimeChannels.Tick.value, str(now))
            self.day_elapsed()

    def day_elapsed(self):
        next_day = self.next_day
        if self.previous_day < next_day:
            self.previous_day = next_day
            self.publish(TimeChannels.NextDay.value, next_day)


class TimeChannels(NoValue):
    NextDay = 'next_day'
    Tick = 'tick'


@click.command()
@click.argument('interval', nargs=1, required=False, default='1s')
def main(interval: str):
    time_emitter = TimeEmitter(interval)
    time_emitter.start()


if __name__ == '__main__':
    main()
