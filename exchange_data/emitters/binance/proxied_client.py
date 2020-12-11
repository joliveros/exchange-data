import requests
from binance.client import Client
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from exchange_data.proxies_base import ProxiesBase


class ProxiedClient(Client, ProxiesBase):
    _proxies = None

    def __init__(self, proxies=None, **kwargs):
        self._proxies = proxies
        ProxiesBase.__init__(self, **kwargs)
        super().__init__(**kwargs)

    @property
    def proxies(self):
        if self._proxies:
            return self._proxies
        else:
            proxy = self.valid_proxies.random_sample()[0]

            return dict(
                http=proxy,
                https=proxy
            )

    def _init_session(self):
        session = requests.session()

        retries = Retry(total=24,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])

        session.mount('http://', HTTPAdapter(max_retries=retries))

        session.proxies.update(self.proxies)
        session.headers.update({'Accept': 'application/json',
                                'User-Agent': 'binance/python',
                                'X-MBX-APIKEY': self.API_KEY})
        return session
