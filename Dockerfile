FROM continuumio/miniconda

ENV NAME exchange-data

WORKDIR /src

RUN apt-get update && apt-get install -y build-essential

COPY environment.yml .
COPY .bashrc /root

RUN bash -c "conda env create -f environment.yml"

COPY . .

CMD ["./exchange-data"]
