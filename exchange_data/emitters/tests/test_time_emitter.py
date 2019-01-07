from datetime import datetime, timezone

import alog
from mock import MagicMock

from exchange_data.emitters import TimeEmitter


class TestTimeEmitter(object):

    def test_next_day_timestamp(self):
        time_emitter = TimeEmitter()
        next_day = time_emitter.next_day

        next_date = datetime.fromtimestamp(next_day/1e3, tz=timezone.utc)

        assert next_date.hour == 0
        assert next_date.minute == 0
        assert next_date.microsecond == 0

    def test_day_elapsed(self, mocker):
        publish_mock: MagicMock = mocker.patch.object(TimeEmitter, 'publish')

        time_emitter = TimeEmitter()
        previous_day = datetime.fromtimestamp(time_emitter.next_day/1e3, tz=timezone.utc)
        time_emitter.previous_day = \
            previous_day.replace(day=previous_day.day - 1).timestamp() * 1000

        time_emitter.day_elapsed()

        publish_mock.assert_called_with('next_day', time_emitter.next_day)


