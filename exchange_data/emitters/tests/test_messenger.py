from exchange_data.emitters import Messenger
from exchange_data.utils import NoValue
from mock import MagicMock

import alog
import pytest


class TestMessenger(object):

    def test_sub_accepts_list_of_enums(self, mocker):

        class Channels(NoValue):
            Tick = 'tick'

        mocker.patch('exchange_data.emitters.messenger.Redis')

        messenger = Messenger()
        messenger.pubsub = MagicMock()

        messenger.sub([Channels.Tick])
        subscribe_mock: MagicMock = messenger._pubsub.subscribe
        subscribe_mock.assert_called_with(['tick'])

    def test_sub_accepts_list_strings(self, mocker):
        mocker.patch('exchange_data.emitters.messenger.Redis')

        messenger = Messenger()
        messenger.pubsub = MagicMock()

        messenger.sub(['tick'])
        subscribe_mock: MagicMock = messenger._pubsub.subscribe
        subscribe_mock.assert_called_with(['tick'])
