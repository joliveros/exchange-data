from exchange_data import settings
from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet
from urllib.parse import urlparse

import alog

alog.set_level(settings.LOG_LEVEL)


class Database(InfluxDBClient):
    def __init__(
        self,
        ssl=False,
        database_name=None,
        influxdb: str = None,
        **kwargs
    ):
        self.connection_str = influxdb if influxdb else settings.DB
        conn_params = urlparse(self.connection_str)

        database = conn_params.path[1:]

        if len(database) == 0:
            if database is None:
                raise Exception('database name required')
            database = database_name

        # alog.info(database)
        # raise Exception()

        netlocs = conn_params.netloc.split(',')
        netloc = netlocs[0]
        parsed_netloc = self.parse_netloc(netloc)

        super().__init__(
            host=parsed_netloc['host'],
            port=parsed_netloc['port'],
            username=parsed_netloc['username'],
            password=parsed_netloc['password'],
            database=database,
            ssl=ssl,
            verify_ssl=settings.CERT_FILE
        )

    def parse_netloc(self, netloc):
        info = urlparse("http://%s" % (netloc))
        return {'username': info.username or None,
                'password': info.password or None,
                'host': info.hostname or 'localhost',
                'port': info.port or 8086}

    def query(self, query: str, *args, **kwargs) -> ResultSet:
        alog.debug(query)

        return super().query(
            database=self.database_name,
            query=query,
            epoch='ms',
            params={'precision': 'ms'},
            chunked=True,
            *args, **kwargs)
