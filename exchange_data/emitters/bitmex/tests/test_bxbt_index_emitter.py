from datetime import datetime

import alog

from exchange_data.emitters.bitmex._bxbt_index_emitter import BXBTIndexEmitter

import dateutil
import mock
import pytest


class TestBXBTIndexEmitter(object):

    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.Messenger')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.BXBTIndexEmitter.on')
    @pytest.mark.vcr()
    def test_init(self, messenger_mock, on_mock):
        emitter = BXBTIndexEmitter(interval='1m')

        assert emitter.channel == '.BXBT_1m'

    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.Messenger')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.BXBTIndexEmitter.publish')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.BXBTIndexEmitter.on')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.SignalInterceptor')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.Database.write_points')
    @pytest.mark.vcr()
    def test_fetch_price(self, messenger_mock, publish_mock, setup_signals_mock,
                         write_points_mock, on_mock):
        emitter = BXBTIndexEmitter(interval='1m', influxdb='http://jose:jade121415@0.0.0.0:28953/')
        start_time = dateutil.parser.parse('2019-03-13 00:19:30.089369')
        emitter.fetch_price('', start_time)

