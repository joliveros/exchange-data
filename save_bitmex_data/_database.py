from influxdb import InfluxDBClient
from urllib.parse import urlparse
from . import settings
import alog


class Database(InfluxDBClient):
    def __init__(self, ssl=False):
        conn_params = urlparse(settings.INFLUX_DB)
        alog.debug(conn_params)
        database = conn_params.path[1:]
        netlocs = conn_params.netloc.split(',')
        netloc = netlocs[0]
        parsed_netloc = self.parse_netloc(netloc)

        super().__init__(host=parsed_netloc['host'],
                         port=parsed_netloc['port'],
                         username=parsed_netloc['username'],
                         password=parsed_netloc['password'],
                         database=database,
                         ssl=ssl,
                         verify_ssl=settings.CERT_FILE)

    def parse_netloc(self, netloc):
        info = urlparse("http://%s" % (netloc))
        return {'username': info.username or None,
                'password': info.password or None,
                'host': info.hostname or 'localhost',
                'port': info.port or 8086}
