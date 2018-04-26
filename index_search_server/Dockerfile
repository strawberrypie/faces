FROM python:3

RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/tmp
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip3 install -r requirements.txt

COPY . /usr/src/app

EXPOSE 8080

ENTRYPOINT ["python3"]

CMD ["-m", "index_search_server"]