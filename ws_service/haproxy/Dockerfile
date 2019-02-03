FROM haproxy:1.9

RUN apt-get update -y && apt-get install -y dnsutils curl

COPY ./haproxy.cfg /usr/local/etc/haproxy/haproxy.cfg
