from redis import Redis
from redis_collections import Set

from exchange_data import settings


class ProxiesBase(object):
  def __init__(self, **kwargs):
    self.redis_client = Redis(host=settings.REDIS_HOST)

  @property
  def valid_proxies(self):
    return Set(key='valid_proxies', redis=self.redis_client)
