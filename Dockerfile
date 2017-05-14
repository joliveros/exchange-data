FROM python:3.6.1-slim

ENV NAME save-bitmex-data

RUN mkdir /code

WORKDIR /code

ADD . /code/

RUN pip install . --no-cache-dir

CMD save-bitmex-data "XBTUSD"
