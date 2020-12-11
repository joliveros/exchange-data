#!/usr/bin/env python
import re

from exchange_data import settings
from exchange_data.emitters.binance.proxied_client import ProxiedClient
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
    def __init__(self, num_tests_per_proxy, min_ratio, **kwargs):
        super().__init__(**kwargs)
        self.num_tests_per_proxy = num_tests_per_proxy
        self.min_ratio = min_ratio

        self.proxies_queue.update(set(self.get_proxies()['hosts']))
        self.proxies_queue.update(self.get_free_proxy_list())
        self.proxies_queue.update(self.valid_proxies)

        results_table = dict()

        for p in self.proxies_queue:
            results_table[p] = 0

        tests = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while len(self.proxies_queue) > 0:
                proxy = self.proxies_queue.pop()

                for t in range(0, self.num_tests_per_proxy):
                    tests.append(executor.submit(test_proxy, proxy))

        for test in tests:
            test_result = test.result()
            if test_result[-1]:
                proxy = test_result[0]
                results_table[proxy] += 1

        results = [r[0] for r in results_table.items()
                   if r[-1] / self.num_tests_per_proxy >= self.min_ratio]

        for p in results:
            self.valid_proxies.add(p)

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

    @staticmethod
    @cache.cache(ttl=60 * 60)
    def get_free_proxy_list_request():
        url = 'https://free-proxy-list.net'
        with requests.session() as session:
            res = session.get(url)
            return res.text

    def get_free_proxy_list(self):
        res = self.get_free_proxy_list_request()

        return re.findall('\d{1,3}\.\d{1,3}.\d{1,3}.\d{1,3}\:\d{1,6}', res)


@click.command()
@click.option('--num-tests-per-proxy', '-t', type=int, default=2)
@click.option('--min-ratio', '-r', type=float, default=0.5)
def main(**kwargs):
    proxies = Proxies(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
