#!/usr/bin/env python

from exchange_data import settings
from exchange_data.emitters.binance._depth_emitter import ProxiedClient
from redis import Redis
from redis_cache import RedisCache
from redis_collections import Set

import alog
import click
import concurrent
import requests
import signal

from exchange_data.proxies_base import ProxiesBase

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


def test_proxy(proxy):
  proxies = dict(
    http=proxy,
    https=proxy
  )

  try:
    ProxiedClient(proxies)
    alog.info(f'## good proxy {proxy} ##')
    return (proxy, True)
  except Exception as e:
    alog.info(e)
    return (proxy, False)


class Proxies(ProxiesBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.proxies_queue.update(set(self.get_proxies()['hosts']))
    self.proxies_queue.update(self.valid_proxies)

    tests = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
      while len(self.proxies_queue) > 0:
        proxy = self.proxies_queue.pop()
        tests.append(executor.submit(test_proxy, proxy))

    for test in tests:
      test_result = test.result()
      if test_result[-1]:
        proxy = test_result[0]
        self.valid_proxies.add(proxy)
        alog.info(f'### valid {proxy} ###')
      else:
        if proxy in self.valid_proxies:
          self.valid_proxies.remove(proxy)
          alog.info(f'### remove {proxy} ###')

    alog.info(len(self.valid_proxies))

  @property
  def proxies_queue(self):
    return Set(key='proxies_queue', redis=self.redis_client)

  @staticmethod
  @cache.cache(ttl=60 * 60)
  def get_proxies():
    with requests.session() as session:
      res = session.get(
        'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all')

      hosts = [host.strip() for host in res.text.split('\r\n')]
      hosts = [host for host in hosts if len(host) > 0]
      alog.info(hosts)
      return dict(hosts=hosts)


@click.command()
def main(**kwargs):
  proxies = Proxies(**kwargs)


if __name__ == '__main__':
  signal.signal(signal.SIGINT, lambda *args: exit(0))
  signal.signal(signal.SIGTERM, lambda *args: exit(0))
  main()
