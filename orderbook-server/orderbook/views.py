import pickle
from multiprocessing import Lock
from time import sleep

import alog

from .apps import OrderbookConfig
from django.apps import apps
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods


@require_http_methods(["GET"])
def index(*args):
    orderbook_app: OrderbookConfig = apps.get_app_config('orderbook')

    try:
        lock = orderbook_app.orderbook_lock
        orderbook = orderbook_app.orderbook

        lock.acquire()
        alog.info(orderbook)
        orderbook_data = {key: orderbook.__dict__[key]
                          for key in orderbook.serialize_keys}

        alog.info(alog.pformat(orderbook_data))

        orderbook_bytes = pickle.dumps(orderbook_data)
        lock.release()
    except Exception as e:
        return HttpResponse(e, status=500)

    return HttpResponse(orderbook_bytes, content_type="application/octet-stream")
