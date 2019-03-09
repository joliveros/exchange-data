from datetime import timedelta, datetime
from dateutil import parser, tz
from exchange_data.streamers._bitmex import BitmexStreamer

import alog
import mock
import numpy as np
import pytest


class TestBitmexStreamer(object):
    start_date = parser.parse('2019-03-06 22:00:03.910369+00:00')
    min_date = parser.parse('2019-03-06 21:46:42.633000+00:00')

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_first_date_available(self, sig_mock):
        end_date = self.start_date + timedelta(seconds=2)

        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=end_date
        )

        assert streamer.min_date == self.min_date

        orderbook_frames = streamer.orderbook_frame_query()
        last_timestamp = None

        for frame in orderbook_frames.get_points(streamer.channel_name):
            last_timestamp = \
                datetime.fromtimestamp(frame['time'] / 1000, tz=tz.tzutc())

        assert end_date > last_timestamp > self.start_date

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.BitmexStreamer.now',
        return_value=start_date
    )
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_init_with_window_size(self, mock_start, sig_mock):
        streamer = BitmexStreamer(
            window_size='2s'
        )
        assert streamer.min_date == self.min_date

        orderbook_frames = streamer.orderbook_frame_query()

        last_timestamp = None
        frame_count = 0

        for frame in orderbook_frames.get_points(streamer.channel_name):
            last_timestamp = \
                datetime.utcfromtimestamp(frame['time'] / 1000)\
                .replace(tzinfo=tz.tzutc())

            frame_count += 1

        window_size = streamer.end_date - streamer.start_date

        assert window_size.seconds == 2
        assert frame_count == 2
        assert streamer.start_date < last_timestamp < streamer.end_date

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_compose_window(self, sig_mock):
        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=self.start_date + timedelta(seconds=2),
        )

        time, index, orderbook = streamer.compose_window()

        assert time.shape == (2,)
        assert index.shape == (2,)
        assert orderbook.shape == (2, 2, 2, 10)

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_one_second_window(self, sig_mock):
        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=self.start_date + timedelta(seconds=1),
        )

        time, index, orderbook = streamer.compose_window()

        assert time.shape == (1,)
        assert index.shape == (1,)
        assert orderbook.shape == (1, 2, 2, 10)

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_ensure_buy_side_is_flipped(self, sig_mock):
        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=self.start_date + timedelta(seconds=1),
        )

        time, index, orderbook = streamer.compose_window()

        book_frame = orderbook[0]
        best_ask = book_frame[0][0][0]
        best_bid = book_frame[1][0][0]

        assert best_ask == 3845.5
        assert best_bid == 3845.0
        assert best_ask - best_bid == 0.5

        assert time.shape == (1,)
        assert index.shape == (1,)
        assert orderbook.shape == (1, 2, 2, 10)

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_generating_exceeds_window_size(self, sig_mock):
        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=self.start_date + timedelta(seconds=1),
        )

        for number in range(5):
            time, index, orderbook = next(streamer)
            orderbook_ar = np.array(orderbook)

            assert index > 0.0
            assert orderbook_ar.shape == (2, 2, 10)

