from datetime import timedelta

from dateutil import parser

from exchange_data.emitters.bitmex._playback_emitter import OrderBookPlayBack

import alog
import pytest


class TestOrderBookPlayBack(object):

    # @pytest.mark.vcr(record_mode='all')
    def test_playback(self):
        orderbook = OrderBookPlayBack(query_interval=2, depths=[21])

        orderbook.run()

    @pytest.mark.vcr()
    def test_get_empty_ranges(self):
        start_date = parser.parse('2018-06-02 22:49:31.148000+00:00')
        end_date = parser.parse('2019-03-14 04:58:26.117528+00:00')

        orderbook = OrderBookPlayBack(
            query_interval=2,
            depths=[21],
            min_date=start_date,
            start_date=start_date,
            end_date=end_date
        )

        ranges = orderbook.get_empty_ranges(
            60000000, '30d',
            start_date=start_date, end_date=end_date
        )

        depth, dt, dt1 = ranges[0]

        diff = dt1 - dt

        assert diff.days == 29
