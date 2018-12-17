FROM continuumio/miniconda

ENV NAME exchange-data

WORKDIR /src

RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /usr/share/man/* && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/lib/dpkg/* && \
    rm -rf /var/lib/dpkg/info* && \
    rm -rf /var/log/*

COPY environment.yml .
COPY .bashrc /root

RUN bash -c "conda env create -f environment.yml" && \
    rm -rf /root/.cache/* && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf /usr/share/man/* && \
    rm -rf /var/lib/dpkg/* && \
    rm -rf /var/lib/dpkg/info* && \
    rm -rf /var/log/* && \
    find / | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

COPY . .

CMD ["./exchange-data"]
